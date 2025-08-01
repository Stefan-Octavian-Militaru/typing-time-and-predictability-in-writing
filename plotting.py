import matplotlib.pyplot as plt
import numpy as np
import csv
from wordfreq import word_frequency
f = open("results.csv", "r")
reader = csv.DictReader(f)
words = []
raw_words = []
lengths = []
writing_times = []
log_probs = []
delays = []
for line in reader:
    words.append(line['word'])
    raw_words.append(line['raw_word'])
    try:
        writing_times.append(float(line['writing_time']))
        lengths.append(float(line['length']))
        log_probs.append(float(line['log_prob']))
        delays.append(float(line['delay']))
    except (ValueError, TypeError):
        continue
lengths = np.array(lengths)
log_probs = np.array(log_probs)
writing_times = np.array(writing_times)
log_func = -1 * log_probs * writing_times / lengths
adjusted_times = writing_times + np.array([0] + delays[:-1])
frequencies = [word_frequency(word, 'en') for word in words]

print(np.corrcoef(writing_times, log_probs)[0, 1])
print(np.corrcoef(writing_times, lengths)[0, 1])
print(np.corrcoef(writing_times, log_func)[0, 1])
print(np.corrcoef(lengths, log_probs)[0, 1])
print(np.corrcoef(lengths, adjusted_times)[0, 1])
print(np.corrcoef(lengths, log_func)[0, 1])
print(np.corrcoef(adjusted_times, log_probs)[0, 1])
print(np.corrcoef(adjusted_times, log_func)[0, 1])
print(np.corrcoef(log_probs, log_func)[0, 1])
print(np.corrcoef(frequencies, writing_times)[0, 1])
print(np.corrcoef(frequencies, adjusted_times)[0, 1])
print(np.corrcoef(frequencies, lengths)[0, 1])
print(np.corrcoef(frequencies, log_probs)[0, 1])
print(np.corrcoef(frequencies, log_func)[0, 1])


# plt.figure(figsize=(10, 6))
# plt.scatter(log_probs, writing_times, alpha=0.3, s=10, edgecolors='none')
# plt.xlabel("Temperature")
# plt.ylabel("Writing Time (ms or s)")
# plt.title("Writing Time vs Temperature")
# plt.grid(True)
# plt.tight_layout()
# plt.show()