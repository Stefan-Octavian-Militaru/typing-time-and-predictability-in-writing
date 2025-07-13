import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import spacy
import torch.nn.functional as F
import threading
import numpy as np
import csv
def get_log_prob_for_mask(model, tokenizer, sentence, original_token):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    log_probs = F.log_softmax(logits[0, masked_index], dim=-1)
    token_id = tokenizer.convert_tokens_to_ids(original_token)
    if token_id == tokenizer.unk_token_id:
        return None
    log_prob = log_probs[0, token_id].item()
    return log_prob
def read_sentences(file):
    session = json.loads(file.read())
    prompts = []
    responses = []
    words = []
    nr_words = 0
    for response in session["prompt_responses"].values():
        prompts.append(response["prompt"])
        responses.append(response["response"])
        sen_words = []
        for word in response["word_metrics"]:
            if not word["was_deleted"]:
                sen_words.append([word["word"], word["word_length"], word["time_taken_seconds"]])
                nr_words += 1
        words.append(sen_words)
    return prompts, responses, words, nr_words
#function that aggregates results from all the different threads and keeps them in order
def thread_worker(model, tokenizer, sentence, original_token, index, results, times, writing_time):
    log_prob = get_log_prob_for_mask(model, tokenizer, sentence, original_token)
    results[index] = log_prob
    times[index] = writing_time

file = open("merged.json")
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased",
)
model = AutoModelForMaskedLM.from_pretrained(
    "google-bert/bert-base-uncased",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
nlp = spacy.load("en_core_web_sm")
prompts, responses, words, nr_words = read_sentences(file)
file.close()
threads = []
results = [None] * nr_words
times = [None] * nr_words
index = 0
for i in range(len(prompts)):
    resp = responses[i]
    prompt = prompts[i]
    #char_count is usefull for determining the position of the masked word within the sentence
    char_count = 0
    #splitting the sentence into words with spacy
    doc = nlp(resp)
    tokens = [token for token in doc if token.pos_ != "PUNCT" and token.pos_ != "SPACE"]
    for j, token in enumerate(tokens):
        #the length and writing time are taken from the json file
        print(token.text, words[i][j][0])
        word_text = token.text
        word_length = words[i][j][1]
        writing_time = words[i][j][2]
        subtokens = tokenizer.tokenize(word_text)
        if not subtokens:
            char_count += word_length + 1
            continue
        first_token = subtokens[0]
        masked_sentence = prompt + resp[:char_count] + tokenizer.mask_token
        #masked_sentence = prompt + resp[:char_count] + tokenizer.mask_token + resp[char_count + word_length:]
        t = threading.Thread(target=thread_worker, args=(
        model, tokenizer, masked_sentence, first_token, index, results, times, writing_time,))
        threads.append(t)
        index += 1
        char_count += word_length + 1
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
f = open("results.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["word", "writing_time", "log_prob"])
cnt = 0
for sentence in words:
    for word in sentence:
        writer.writerow([word[0], word[2], results[cnt]])
        cnt += 1
f.close()
results = [r for r in results if r is not None]
times = [t for t in times if t is not None]
corr = np.corrcoef(times, results)[0, 1]
print(f"Correlation between writing time and surprisal scores: {corr}")
