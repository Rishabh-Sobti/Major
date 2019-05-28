import csv

import os
won = list(csv.reader(open('won.csv')))
contested= list(csv.reader(open('contested.csv')))
final = []
def state_to_standard(fname):
    switcher = {
        'Jharkhand':14,
        'Karnakata':15,
        'Himachal':12,
        'Gujarat':10,
        'Goa':9,
        'Kerela':16,
        'JK':13,
        'Haryana':11,
        'lse':33,
        'Telangana':28,
        'WB':32,
        'UP':30,
        'TN':27,
        'Sikkim':26,
        'UK':31,
        'Rajasthan':25,
        'Tripura':29,
        'dmk':39,
        'bjp':2,
        'BJP':2,
        'shivsena':10,
        'pmk':27,
        'inc':4,
        'tmc':7,
        'aimim':30,
        'NCP':14,
        'IUML':33,
        'iuml':33,
        'congress':4,
        'aiadmk':3,
        'aiudf':22,
        'JD(S)':26,
        'PDP':21,
        'jkpdp':21,
        'npf':19,
        'sad':13,
        'ncp':14,
        'bsp':38,
        'BSP':38,
        'cpim':12,
        'sdf':28,
        'CPI(M)':12,
        'akali':13,
        'CPI':35,
        'tdp':6,
        'jd(s)':26,
        'jds':26,
        'cpi':35,
        'cpm':12,
        'sp':20,
        'JMM':29,
        'aap':32,
        'BJP':2,
        'jdu':31,
        'ljp':11,
        'trs':8,
        'other':37,
        'bjd':5,
        'swabhimani':18,
        'INLD':17,
        'INC':4
    }
    return switcher.get(fname, '0')

for name in os.listdir('./targetData'):

    if name.endswith('.ans.csv'):

        fname = name[:-12].split('_')
        state = state_to_standard(fname[0])
        party = state_to_standard(fname[1])
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

            row = [won[state][0], won[0][party], pos, neg, contested[state][party-1], won[state][1], pos+neg, won[state][party]]
            final.append(row)
csv.writer(open('./targetData/final_dataset.csv', 'w'),delimiter=',').writerows(final)

