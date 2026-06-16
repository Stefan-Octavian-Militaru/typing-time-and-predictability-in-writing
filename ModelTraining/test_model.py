import math
import joblib
import pandas as pd
import numpy as np
from wordfreq import zipf_frequency
package = joblib.load("typing_time_model_package.pkl")

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
        return 0
    return total_distance


model = package["model"]
features = package["features"]

example_sentence = "I went to the shop where we had a nice ice cream"
nr_words = len(example_sentence.split(" "))
prev_word_length = 0
sentence_features = []

for word_cnt, word in enumerate(example_sentence.split(" ")):
    sample = {
        'distance_backspace_interaction' : 0 ,
        'edit_length_ratio': 0 ,
        'frequency_length_ratio': 0 ,
        'word_relative_position': word_cnt / nr_words,
        'edit_distance': 0 ,
        'nr_versions': 1 ,
        'keyboard_distance': keyboard_distance(word),
        'backspaces' : 0,
        'word_frequency':  zipf_frequency(word, "en"),
        'prev_word_length': prev_word_length,
        'has_comma': (int) (',' in word),
        'version_edit_interaction': 0 ,
        'was_deleted': 0 ,
        'is_hyphenated': (int) ('-' in word) ,
        'has_period': (int) ('.' in word) ,
        'word_length': len(word) ,
        'word_position': word_cnt
    }

    # sample["version_edit_interaction"] = (
    #         sample["nr_versions"] * sample["edit_distance"]
    # )
    sample["frequency_length_ratio"] = (
            sample["word_frequency"] / (sample["word_length"] + 1)
    )


    # sample["edit_length_ratio"] = (
    #         sample["edit_distance"] / (sample["word_length"] + 1)
    # )
    #
    # sample["distance_backspace_interaction"] = (
    #         sample["keyboard_distance"] * sample["backspaces"]
    # )
    prev_word_length = sample["word_length"]
    sentence_features.append(sample)

word_times = []

for word_features in sentence_features:
    sample_df = pd.DataFrame([word_features])
    sample_df = sample_df[features]


    log_pred = model.predict(sample_df)
    pred_seconds = np.expm1(log_pred)
    word_times.append(pred_seconds[0])

total_sentence_time = sum(word_times)

for word_cnt, word in enumerate(example_sentence.split(" ")):
    print(f"'{word}' took {round(word_times[word_cnt], 2)} seconds to type, approximately {round(word_times[word_cnt] / total_sentence_time * 100, 2)}% of the total time.")

print(f"The sentence took {round(total_sentence_time, 2)} seconds in total")