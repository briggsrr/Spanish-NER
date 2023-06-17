# Spanish-NER


### General overview:

![Spanish NER Diagram](assets/spanish_ner.drawio.png)

[Kaggle](https://www.kaggle.com/competitions/ucsb-cs190i-hw4-ner-on-spanish-data/leaderboard)

##### running inference (this produces test_ans.csv, everything else should be configured)
`python inference.py` 


##### running training:
`python configure.py`
`python train.py`

##### dir breakdown
- assets -> contains diagram 
- spanish_ner -> contains model from iteration 1 of system 
- tokenizer -> contains tokenizer from iteration 1 of system 
- spanish_ner_2 -> contains model from iteration 2 of system 
- tokenizer_2 -> contains tokenizer from iteration 2 of system 
- clean.py -> postprocessing script (NOT USED IN FINAL, HURT PERFORMANCE)
- configure.py -> creates transformed_train.csv and transformed_val.csv loaded in train.py
- train.py -> used to fine tune BERT models on spanish dataset 

# Spanish-NER
