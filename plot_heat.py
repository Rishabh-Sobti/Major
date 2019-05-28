import seaborn as sns
import csv
import matplotlib.pyplot as plt
data = list(csv.reader(open('final_dataset.csv')))
states = list(set([row[0] for row in data]))
parties = list(set([row[1] for row in data]))
import numpy as np
corr = np.zeros((len(states), len(parties)))
for row in data:
    i=0
    j=0
    for k in range(len(states)):
        if(row[0]==states[k]):
            i=k
            break
    for l in range(len(parties)):
        if(row[1]==parties[l]):
            j=l
            break
    corr[i][j]=float(row[2])/float(row[6])
import pandas as pd
corr = pd.DataFrame(corr, states, parties)
fig, ax = plt.subplots()
ax = sns.heatmap(corr)
plt.show()
