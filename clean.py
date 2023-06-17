#take in test_ans_pre.csv and creates test_ans.csv

#data post processing: 
#{'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}

#if _ then next should be _ 
#04
#15
#26
#37

#Post processing rules: 
#1. I- B- fixing?:
#reach a 4 and there is not a 0 or 4 before, turn 4->0
#reach a 5 and there is not a 1 or 5 before, turn 5->1
#reach a 6 and there is not a 2 or 6 before, turn 6->2
#reach a 7 and there is not a 3 or 7before, turn 7->3

#2. gap fixing?: 
#if there is a 2 -> 8 -> 6 turn 8 -> 6
#if there is a 1 -> 8 -> 5 turn 8 -> 5
#if there is a 0 -> 8 -> 4 turn 8 -> 4
#if there is a 3 -> 8 -> 7 turn 8 -> 7
#if there is 4 8 4 make the 8 a 4
#if there is a 5 8 5 make the 8 a 5
#if there is a 6 8 6 make the 8 a 6
#if there is a 7 8 7 make the 8 a 7

import csv
import pandas as pd

# Load the data
data = pd.read_csv('test_ans_pre.csv')

# Create a list from the 'label' column
labels = data['label'].tolist()

# Apply rule 1
# for i in range(1, len(labels)):
#     if labels[i] == 4 and labels[i-1] not in [0, 4]:
#         labels[i-1] = 0
#     elif labels[i] == 5 and labels[i-1] not in [1, 5]:
#         labels[i-1] = 1
#     elif labels[i] == 6 and labels[i-1] not in [2, 6]:
#         labels[i-1] = 2
#     elif labels[i] == 7 and labels[i-1] not in [3, 7]:
#         labels[i-1] = 3

# # Apply rule 2
for i in range(1, len(labels) - 1):
    if labels[i] == 8:
        if labels[i-1] == 2 and labels[i+1] == 6:
            labels[i] = 6
        elif labels[i-1] == 1 and labels[i+1] == 5:
            labels[i] = 5
        elif labels[i-1] == 0 and labels[i+1] == 4:
            labels[i] = 4
        elif labels[i-1] == 3 and labels[i+1] == 7:
            labels[i] = 7
        elif labels[i-1] == 4 and labels[i+1] == 4:
            labels[i] = 4
        elif labels[i-1] == 5 and labels[i+1] == 5:
            labels[i] = 5
        elif labels[i-1] == 6 and labels[i+1] == 6:
            labels[i] = 6
        elif labels[i-1] == 7 and labels[i+1] == 7:
            labels[i] = 7

# Update the 'label' column
data['label'] = labels

# Save the data
data.to_csv('test_ans.csv', index=False)

