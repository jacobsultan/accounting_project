#Accounting_Project

The data for this project has been anonymized for obvious security reasons.

The data_preprocessing.ipynb file cleans the data outputting proessed_data.csv which is then explored in data_exploration.ipynb

other_models.ipynb investigates which ML models would be best used for categorising the data and uses the model_evaluator.py function in the functions folder.

The final evaluations are conducted in ensemble.ipynb which also uses a custom built llm from scratch found in own_llm.ipynb and llm_classes folder.

** Pls note that the accuracy on this dataset of the ensemble was commonly within a percentage of a perfect score, however on anonymising the data this fell to around 90% **

