import csv
import os
won = list(csv.reader(open('won.csv')))
states = []
parties= []
for name in os.listdir('./targetData'):

    if name.endswith('.ans.csv'):

        fname = name[:-12].split('_')
        try:
            states.append(fname[0])
            parties.append(fname[1])
        except:
            print(fname)
print("###########----------------------------########")
print("Your states: ")
states = (list(set(states)))
for x in states:
    print("        '"+x+"':,")
print("####----------------------------###############")
print("Your parties: ")
parties = (list(set(parties)))
for x in parties:
    print("        '"+x+"':,")
print("#######------------------------############")
print("State list with number:")
first = 0
for row in won:
    for x in row:
        if(first==0):
            first=1
            break
        else:
            print(x+" : "+str(first))
            first+=1
            break
print("###############----------------------####")
print("Party list with number:")
first = 0
for row in won:
    for x in row:
        if(first==0 or first==1):
            first+=1
        else:
            print(x+" : "+str(first))
            first+=1
    break



