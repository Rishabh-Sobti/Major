import csv
def getTweetsRawData( fileName ):
    # read all tweets and labels
    fp = open( fileName, 'rb' )
    reader = csv.reader( fp, delimiter=';', quotechar='"', escapechar='\\' )
    tweets = []
    first = 0
    for row in reader:
        if (first==0):
            first=1
            continue
        try:
            tweets.append( [row[4], '', row[10], row[10] ] )
        except:
            continue

    return tweets # 0: Text # 1: class # 2: subject # 3: query
