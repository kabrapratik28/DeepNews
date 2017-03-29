# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
inp_file = pd.read_csv("first500.txt",sep=" ",header=None)

hindi_word = inp_file[0]
inp_file.drop(0,axis=1,inplace=True)

import numpy as np
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
reduced= model.fit_transform(inp_file) 

for i in range(len(hindi_word)):
    hindi_word[i] = unicode(hindi_word[i], "utf-8")
    
for i in range(len(hindi_word)):
    hindi_word[i] = hindi_word[i].encode("utf-8","ignore")
    
x = [obj[0]*10000 for obj in reduced]
y = [obj[1]*10000 for obj in reduced]


import csv

csvfile = "scatter-label.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in x:
        writer.writerow([val]) 
        
csvfile = "scatter-labelY.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in y:
        writer.writerow([val])

matr = []
for word in hindi_word:
    matr.append(word)

thefile = open('words.txt', 'w')
for item in matr:
  thefile.write("%s\n" % item)
