import json
from transformers import pipeline, AutoModelForTokenClassification, BertTokenizerFast
import pandas as pd
import numpy as np
import random
import torch
import csv
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(420)

# Define the label list
label_list = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']

# Create id2label and label2id mappings
id2label = {str(i): label for i, label in enumerate(label_list)}
label2id = {label: str(i) for i, label in enumerate(label_list)}

#Load and update the model configuration
config = json.load(open("spanish_ner_2/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("spanish_ner_2/config.json", "w"))

#Load the model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("spanish_ner_2")
tokenizer = BertTokenizerFast.from_pretrained("tokenizer_2")

def load_sentences_from_csv(filepath):
    test_data = pd.read_csv(filepath, skip_blank_lines=False)
    sentences = []
    sentence = []
    for _, row in test_data.iterrows():
        if pd.isna(row['word']):  # check for nan
            #print(sentence)
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(row['word'])
    if sentence:  #last sentence if not empty
        sentences.append(sentence)

    sentences = [' '.join(sentence) for sentence in sentences]

    return sentences

sentences = load_sentences_from_csv('test_noans.csv')

def tokenize_and_get_word_ids(sentence): 
    new_sentence = sentence.split()
    tokenized_inputs = tokenizer(new_sentence, truncation=True, is_split_into_words=True) 
    word_ids = tokenized_inputs.word_ids()

    # Assign None to subtokens
    previous_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id == previous_word_id:
            word_ids[i] = None
        previous_word_id = word_id

    return tokenized_inputs, word_ids

sentences = load_sentences_from_csv('test_noans.csv')
master_id = 0

with open('test_ans.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'label'])  
    for sentence in sentences:

        inputs, word_ids = tokenize_and_get_word_ids(sentence)

        #print(word_ids)
        inputs = {k: torch.tensor([v]) for k, v in inputs.items()}  # Convert to tensors

        #Get  predictions
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

        # Map the predictions back to the original words
        for word_id, prediction in zip(word_ids, predictions):
            if word_id is not None:  
                writer.writerow([master_id, prediction])
                master_id += 1
