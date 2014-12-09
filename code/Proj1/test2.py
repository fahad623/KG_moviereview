import pandas as pd


#df = pd.DataFrame({'col1':[2, 9, 6], 'col2':[11, 12, 13]})
#
#df.sort(['col1'])
#
#print df

import csv
import re

location_train = "../../data/train.tsv"
location_test = "../../data/test.tsv"


def clean(s):
    return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
  
  

with open(location_train) as infile, open('filename1.txt', 'wb') as outfile:
    reader = csv.DictReader(infile, delimiter="\t")
    fieldnames = ['PhraseId',	'SentenceId',	'Phrase',	'Sentiment']
    writer = csv.DictWriter(outfile, fieldnames = fieldnames)
    writer.writeheader()
    for row in reader:
        row['Phrase'] = clean(row['Phrase'])
        count = row['Phrase'].count(" ")+1
        
        if count == 1:            
            writer.writerow(row)
        else count ==2;
            
        
        