import torch
from torch import cosine_similarity
from transformers import AutoTokenizer, BertModel
import csv

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

tokenizer.add_special_tokens({'additional_special_tokens': ['<MISSING>']})
model.resize_token_embeddings(len(tokenizer))


def get_embeddings(model, encoding):
    with torch.no_grad():
        outputs = model(**encoding)
    return outputs.last_hidden_state, outputs


def compare_embeddings(sentence, word_char_start, word_char_end, model, tokenizer):
    encoding = tokenizer(
        sentence,
        return_tensors='pt',
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True
    )
    offsets = encoding["offset_mapping"][0]
    input_ids = encoding["input_ids"][0]
    token_start = None
    token_end = None
    for i, (start, end) in enumerate(offsets):
        if start == end == 0:
            continue
        if start <= word_char_start < end:
            token_start = i
        if start < word_char_end <= end:
            token_end = i + 1
            break

    if token_start is None or token_end is None:
        return None, None

    span_length = token_end - token_start
    modified_ids = input_ids.tolist()
    replacement_id = tokenizer.convert_tokens_to_ids("<MISSING>")
    modified_ids[token_start:token_end] = [replacement_id] * span_length

    modified_encoding = {
        "input_ids": torch.tensor([modified_ids]),
        "attention_mask": torch.ones_like(torch.tensor([modified_ids]))
    }

    encoding.pop("offset_mapping")
    modified_encoding.pop("offset_mapping", None)
    full_embed, outputs = get_embeddings(model, encoding)
    mod_full_embed, mod_outputs = get_embeddings(model, modified_encoding)
    full_sentence_emb = full_embed[0, 1:-1, :].mean(dim=0).unsqueeze(0)
    mod_sentence_emb = mod_full_embed[0, 1:-1, :].mean(dim=0).unsqueeze(0)

    full_sim = cosine_similarity(full_sentence_emb, mod_sentence_emb).item()
    exclude_indices = list(range(1, token_start)) + list(range(token_end, len(input_ids)-1))
    excl_full_emb = full_embed[0, exclude_indices, :].mean(dim=0).unsqueeze(0)
    excl_mod_emb = mod_full_embed[0, exclude_indices, :].mean(dim=0).unsqueeze(0)

    excl_sim = cosine_similarity(excl_full_emb, excl_mod_emb).item()

    return full_sim, excl_sim


file = open("similarities.csv", "w", newline="")
sentence_file = open("sentences.txt")
sentences = sentence_file.readlines()
writer = csv.writer(file)

writer.writerow(["word", "full sentence similarity", "similarity excluding word"])

for sentence in sentences:
    sentence = sentence.strip("\n")
    words = sentence.split()
    char_index = 0
    for word in words:
        stripped = word.strip(".,?!")
        start = sentence.find(stripped, char_index)
        end = start + len(stripped)
        char_index = end + 1

        sim1, sim2 = compare_embeddings(sentence, start, end, model, tokenizer)
        if sim1 is None:
            continue
        writer.writerow([stripped, sim1, sim2])
    writer.writerow(["NEW_SENTENCE", "0", "0"])