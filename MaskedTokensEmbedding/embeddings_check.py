import torch
from torch import cosine_similarity
from transformers import AutoTokenizer, BertModel
import csv

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

tokenizer.add_special_tokens({'additional_special_tokens': ['<MISSING>', '<START_WORD>', '<END_WORD>']})
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
        add_special_tokens=False
    )

    offsets = encoding["offset_mapping"][0]
    input_ids = encoding["input_ids"][0].tolist()


    token_start = None
    token_end = None

    for i, (start, end) in enumerate(offsets):
        if start <= word_char_start < end:
            token_start = i
        if start < word_char_end <= end:
            token_end = i + 1
            break
    if token_start is None or token_end is None:
        print(f"Warning: Could not find word '{stripped}' in sentence")
        return None, None

    if token_start >= len(input_ids) or token_end > len(input_ids):
        print(f"Warning: Token indices out of bounds for '{stripped}'")
        return None, None
    start_marker = tokenizer.convert_tokens_to_ids("<START_WORD>")
    end_marker = tokenizer.convert_tokens_to_ids("<END_WORD>")
    modified_ids = (
        input_ids[:token_start] +
        [start_marker] +
        input_ids[token_start:token_end] +
        [end_marker] +
        input_ids[token_end:]
    )

    input_ids_tensor = torch.tensor([input_ids])
    modified_ids_tensor = torch.tensor([modified_ids])

    with torch.no_grad():
        full_embed = model(input_ids_tensor).last_hidden_state[0]
        mod_full_embed = model(modified_ids_tensor).last_hidden_state[0]

    full_sentence_emb = full_embed.mean(dim=0, keepdim=True)
    mod_sentence_emb = mod_full_embed.mean(dim=0, keepdim=True)

    word_embed = full_embed[token_start:token_end].mean(dim=0, keepdim=True)
    mod_word_embed = mod_full_embed[token_start + 1 : token_end + 1].mean(dim=0, keepdim=True)

    sent_sim = cosine_similarity(full_sentence_emb, mod_sentence_emb).item()
    word_sim = cosine_similarity(word_embed, mod_word_embed).item()

    return sent_sim, word_sim, token_end - token_start


file = open("similarities_modified.csv", "w", newline="")
sentence_file = open("sentences.txt")
sentences = sentence_file.readlines()
writer = csv.writer(file)

writer.writerow(["word", "full sentence similarity", "target word similarity", "word token length"])

for sentence in sentences:
    sentence = sentence.strip("\n")
    words = sentence.split()
    char_index = 0
    for word in words:
        stripped = word.strip(".,?!")
        start = sentence.find(stripped, char_index)
        if start == -1:
            continue
        end = start + len(stripped)
        char_index = end
        sim1, sim2, nr_tokens = compare_embeddings(sentence, start, end, model, tokenizer)
        if sim1 is None:
            continue
        writer.writerow([stripped, sim1, sim2, nr_tokens])
    writer.writerow(["NEW_SENTENCE", "0", "0", "0"])
file.close()
sentence_file.close()