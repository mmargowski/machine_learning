#!/usr/bin/python

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

poi = 0

for i in enron_data:
    if(enron_data[i]["poi"]==1):
        poi += 1
        
print(poi)

