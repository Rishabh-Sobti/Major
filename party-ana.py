import matplotlib.pyplot as plt
import csv
import os

for name in os.listdir('./targetData'):
    if name.endswith('.ans.csv'):
        with open('./targetData/'+name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            first = 0
            pos = 0
            neg = 0
            for row in reader:
                if first==0:
                    first=1
                    continue
                if row[1]=='pos':
                    pos+=1
                elif row[1]=='neg':
                    neg+=1
            labels = 'Positive', 'Negative'
            sizes = [pos, neg]
            print(sizes)
            print(name[:-12])
