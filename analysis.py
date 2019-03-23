import matplotlib.pyplot as plt
import csv
import os

for name in os.listdir('.'):
    if name.endswith('.ans.csv'):
        with open(name) as csv_file:
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
            explode = (0.1, 0)
            fig1, ax1 = plt.subplots()
            title = 'Total tweets found : '+str(pos+neg)
            ax1.set_title(title)
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=['#3498db', '#e74c3c'])
            ax1.axis('equal')
            plt.savefig('analysis/'+name[:-12]+'.png')
        
