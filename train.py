import numpy as np 
from transformers import BertTokenizerFast, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, AdamW
import ast
import csv
import datasets 
import torch 
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(420)

def load_data(file_path):
    token_docs = []
    tag_docs = []

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  

        for row in reader:
            #for consistency parse string representation of list into actual list
            tokens = ast.literal_eval(row[0])
            tags = ast.literal_eval(row[1])

            token_docs.append(tokens)
            tag_docs.append(tags)

    return token_docs, tag_docs

train_tokens, train_tags = load_data('transformed_train.csv')
val_tokens, val_tags = load_data('transformed_val.csv') 

tag_to_id = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}

# Replace the string tags with their corresponding integer labels
train_tags = [[tag_to_id[tag] for tag in doc] for doc in train_tags]
val_tags = [[tag_to_id[tag] for tag in doc] for doc in val_tags]



# tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
tokenizer = BertTokenizerFast.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')


#testing simple example
# example = train_tags[0]
# tokenized_input = tokenizer(example, is_split_into_words=True)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
# word_ids = tokenized_input.word_ids()
# print(word_ids)


#idea from documentation 
def tokenize_and_align_labels(instances, label_all_tokens=False): 
    tokenized_inputs = tokenizer(instances["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(instances["tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        previous_word_idx = None 
        label_ids = []
        for word_idx in word_ids: 
            if word_idx is None: 

                label_ids.append(-100)

            elif word_idx != previous_word_idx:               
                label_ids.append(label[word_idx]) 
            else: 
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs 


# Now you can create the datasets
train_dataset = datasets.Dataset.from_dict({
    "tokens": train_tokens,
    "tags": train_tags
})

val_dataset = datasets.Dataset.from_dict({
    "tokens": val_tokens,
    "tags": val_tags
})

tokenized_datasets = datasets.DatasetDict({
    "train": train_dataset,
    "val": val_dataset
})

tokenized_datasets = tokenized_datasets.map(tokenize_and_align_labels, batched=True)    
# model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)
model = AutoModelForTokenClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", num_labels=9)
#dccuchile/bert-base-spanish-wwm-cased

# args = TrainingArguments(
#     "spanish_ner",
#     num_train_epochs=10,
#     learning_rate=2e-5,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=64,
#     warmup_steps=1000,
#     weight_decay=0.02,
#     evaluation_strategy = "epoch",
#     gradient_accumulation_steps=2,  
# )

args_2 = TrainingArguments(
    "spanish_ner_2",
    num_train_epochs=20,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=1000,
    weight_decay=0.03,
    evaluation_strategy = "epoch",
    gradient_accumulation_steps=2,  
)


data_collator = DataCollatorForTokenClassification(tokenizer)

# trainer = Trainer( 
#     model, 
#     args, 
#     train_dataset=tokenized_datasets["train"], 
#     eval_dataset=tokenized_datasets["val"], 
#     data_collator=data_collator, 
#     tokenizer=tokenizer, 
# ) 

trainer_2 = Trainer( 
    model, 
    args_2, 
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["val"], 
    data_collator=data_collator, 
    tokenizer=tokenizer, 
) 


# trainer.train() 
trainer_2.train()
model.save_pretrained("spanish_ner_2")
tokenizer.save_pretrained("tokenizer_2")
