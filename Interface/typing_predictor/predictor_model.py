import math
import joblib
import pandas as pd
import numpy as np
import os
import wordfreq
import Levenshtein

BASE = os.path.dirname(__file__)
model_path = os.path.join(BASE, "typing_time_model_package.pkl")
package = joblib.load(model_path)

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
            # print(f"One of the following keys is unrecognised: {keyA} {keyB} . Substituting with average key distance of 3.95")
            total_distance += 3.95
        nr_distances += 1
    if nr_distances == 0:
        return 0
    return total_distance


def predict(word_data, actual_time):
    words = [word[0] for word in word_data]
    nr_words = len(words)
    prev_word_length = 0
    sentence_features = []
    for word_cnt, word_info in enumerate(word_data):
        word = word_info[0]
        versions = word_info[2]
        sample = {
            'distance_backspace_interaction' : 0 ,
            'edit_length_ratio': 0 ,
            'frequency_length_ratio': 0 ,
            'word_relative_position': word_cnt / nr_words,
            'edit_distance': Levenshtein.distance(word, versions[0]),
            'nr_versions': len(versions) ,
            'keyboard_distance': 0,
            'backspaces' : word_info[1],
            'word_frequency':  wordfreq.zipf_frequency(word, "en"),
            'prev_word_length': prev_word_length,
            'has_comma': (int) (',' in word),
            'version_edit_interaction': 0 ,
            'was_deleted': 0 ,
            'is_hyphenated': (int) ('-' in word) ,
            'has_period': (int) ('.' in word) ,
            'word_length': len(word) ,
            'word_position': word_cnt
        }
        if len(versions) > 1:
            sample["keyboard_distance"] = keyboard_distance(reconstruct_word(versions, word_info[1]))
        else:
            sample["keyboard_distance"] = keyboard_distance(word)

        sample["version_edit_interaction"] = (
                sample["nr_versions"] * sample["edit_distance"]
        )
        sample["frequency_length_ratio"] = (
                sample["word_frequency"] / (sample["word_length"] + 1)
        )


        sample["edit_length_ratio"] = (
                sample["edit_distance"] / (sample["word_length"] + 1)
        )
        
        sample["distance_backspace_interaction"] = (
                sample["keyboard_distance"] * sample["backspaces"]
        )
        prev_word_length = sample["word_length"]
        sentence_features.append(sample)

    word_times = [[word, 0, 0, 0] for word in words]
    for cnt, word_features in enumerate(sentence_features):
        sample_df = pd.DataFrame([word_features])
        sample_df = sample_df[features]
        log_pred = model.predict(sample_df)
        pred_seconds = np.expm1(log_pred)
        word_times[cnt][1] = pred_seconds[0]

    total_sentence_time = sum([word[1] for word in word_times])
    adjustment_factor = actual_time/total_sentence_time
    for word_time in word_times:
        word_time[2] = word_time[1] * adjustment_factor
        word_time[3] = word_time[1] / total_sentence_time * 100

    return total_sentence_time, word_times


model = package["model"]
features = package["features"]



