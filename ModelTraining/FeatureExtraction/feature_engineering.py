import pandas as pd
import numpy as np
import csv
import math
from wordfreq import zipf_frequency
import Levenshtein

keyboard_coords = {
    "!" : (-0.5, 3),
    "1" : (-0.5, 3),
    "2" : (0.5, 3),
    "3" : (1.5, 3),
    "4" : (2.5, 3),
    "5" : (3.5, 3),
    "6" : (4.5, 3),
    "7" : (5.5, 3),
    "8" : (6.5, 3),
    "9" : (7.5, 3),
    "0" : (8.5, 3),
    "-" : (9.5, 3),
    "=" : (10.5,3),
    #The tilda used here is a substitute for the backspace key
    "~" : (12, 3),

    "q": (0, 2),
    "w": (1, 2),
    "e": (2, 2),
    "r": (3, 2),
    "t": (4, 2),
    "y": (5, 2),
    "u": (6, 2),
    "i": (7, 2),
    "o": (8, 2),
    "p": (9, 2),
    "[": (10, 2),
    "]": (11, 2),
    "\\": (12.2, 2),

    "a": (0.2, 1),
    "s": (1.2, 1),
    "d": (2.2, 1),
    "f": (3.2, 1),
    "g": (4.2, 1),
    "h": (5.2, 1),
    "j": (6.2, 1),
    "k": (7.2, 1),
    "l": (8.2, 1),

    ";": (9.2, 1),
    ":": (9.2, 1),
    "ș": (9.2, 1),
    "'": (10.2, 1),
    "ț": (10.2, 1),

    "z": (0.8, 0),
    "x": (1.8, 0),
    "c": (2.8, 0),
    "v": (3.8, 0),
    "b": (4.8, 0),
    "n": (5.8, 0),
    "m": (6.8, 0),

    ",": (7.8, 0),
    ".": (8.8, 0),
    "/": (9.8, 0),
    "?": (9.8, 0)
}
def longest_common_prefix(word_1, word_2):
    nr_letters = min(len(word_1), len(word_2))
    for index in range(nr_letters):
        if word_1[index] != word_2[index]:
            return index
    return -1
def reconstruct_word(versions, backspaces):
    unique_versions = set([])
    for version in versions:
        if version in unique_versions:
            versions.remove(version)
        unique_versions.add(version)
    previous = versions[0]
    #since the number of backspaces between 2 letters is irrelevant for calculating keyboard
    #distance, we insert just one backspace between the other versions and put the remaining
    #backspace keys after the first version
    rec_word = previous + "~" * (backspaces - len(versions) + 2)
    index = 0
    for current in versions[1:]:
        index = longest_common_prefix(previous, current)
        #tilda is used as a substitute for a backspace
        rec_word += current[index:] + "~"
        previous = current
    rec_word = rec_word.strip("~")
    return rec_word
def keyboard_distance(word):
    total_distance = 0
    nr_distances = 0
    for cnt in range(len(word) - 1):
        keyA = word[cnt].lower()
        keyB = word[cnt + 1].lower()
        try:
            total_distance += math.sqrt ((keyboard_coords[keyA][0] - keyboard_coords[keyB][0]) ** 2 +
                                    (keyboard_coords[keyA][0] - keyboard_coords[keyB][0]) ** 2)
        except KeyError:
            print(f"One of the following keys is unrecognised: {keyA} {keyB} . Substituting with average key distance of 3.95")
            total_distance += 3.95
        nr_distances += 1
    if nr_distances == 0:
        return 0,0
    return total_distance, total_distance / nr_distances
def count_vowels(word):
    vowels = set("aeiou")
    return len([letter for letter in word if letter in vowels])
def read_sentences(file_name):
    df = pd.read_json(file_name)
    rows = []
    for prompt_id, prompt_data in df["prompt_responses"].items():
        word_cnt = 0
        for word_dict in prompt_data["word_metrics"]:
            if not word_dict["was_deleted"] and word_dict["word_length"] > 0:
                if word_dict["time_taken_seconds"] != 0:
                    typing_time = word_dict["time_taken_seconds"]
                else:
                    typing_time = 0.05
                word_cnt += 1
                row = word_dict.copy()
                row["was_deleted"] = int(row["was_deleted"])
                row["prompt_id"] = prompt_id
                row["prompt"] = prompt_data["prompt"]
                row["response"] = prompt_data["response"]
                word = str(word_dict['word'])
                word_length = len(word)
                row["char_count"] = word_length
                row["is_short"] = int(word_length <= 4)
                row["is_long"] = int(word_length >= 8)
                row["chars_per_second"] = word_length / typing_time
                row["vowel_ratio"] = count_vowels(word) / word_length
                row["word_frequency"] = zipf_frequency(word, "en")
                row["has_period"] = int(word[-1] == '.')
                row["has_comma"] = int(word[-1] == ',')
                row["is_hyphenated"] = int(word.count('-') != 0)
                row["nr_versions"] = len(word_dict['versions'])
                if len(rows) != 0:
                    row["prev_word_length"] = rows[-1]["word_length"]
                    row["prev_time"] = rows[-1]["time_taken_seconds"]
                    row["prev_pause"] = rows[-1]["delay_after_word_seconds"]
                    row["prev_pause_ratio"] = rows[-1]["delay_after_word_seconds"] / typing_time
                else:
                    row["prev_word_length"] = 0
                    row["prev_time"] = 0
                    row["prev_pause"] = 0
                    row["prev_pause_ratio"] = 0
                row["word_position"] = word_cnt
                row["word_relative_position"] = word_cnt / len(prompt_data["word_metrics"])
                if len(word_dict['versions']) > 1:
                    row["keyboard_distance"], row["avg_keyboard_distance"] = keyboard_distance(reconstruct_word(word_dict['versions'], word_dict['backspaces']))
                else:
                    row["keyboard_distance"], row["avg_keyboard_distance"] = keyboard_distance(word)
                row["sentence_length"] = len(prompt_data["response"])
                row["edit_distance"] = Levenshtein.distance(word, word_dict["versions"][0])
                #interactive features, obtained from the interaction of different characteristics
                row["distance_backspace_interaction"] = (
                        row["keyboard_distance"] * row["backspaces"]
                )
                #adding 1 to word length to avoid divided by zero error generated by some mistyped words in the dataset
                row["edit_length_ratio"] = (
                        row["edit_distance"] / (row["word_length"] + 1)
                )

                row["frequency_length_ratio"] = (
                        row["word_frequency"] / (row["word_length"] + 1)
                )

                row["position_sentence_interaction"] = (
                        row["word_relative_position"] * row["sentence_length"]
                )

                row["version_edit_interaction"] = (
                        row["nr_versions"] * row["edit_distance"]
                )

                rows.append(row)
    df = pd.DataFrame(rows)
    return df
words = read_sentences("merged.json")
print(words.columns)
words.to_csv("word_features.csv")


