import os
for name in os.listdir('.'):
    if name.endswith('.ans.csv'):
        if not name.endswith('.csv.ans.csv'):
            os.rename(name, name[:-8]+'.csv.ans.csv')
