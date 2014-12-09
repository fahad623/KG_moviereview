import csv
import re

location_train = "../../../data/train.tsv"

location_train_oneword = "../../../data/train_oneword.csv"

location_train_twoword = "../../../data/train_twoword.csv"

location_train_morethantwoword = "../../../data/train_morethantwoword.csv"


def clean(s):
    return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
  
  

with open(location_train) as infile, open(location_train_oneword, 'wb') as outfile1, open(location_train_twoword, 'wb') as outfile2, open(location_train_morethantwoword, 'wb') as outfile3:
    reader = csv.DictReader(infile, delimiter="\t")
    fieldnames = ['PhraseId',	'SentenceId',	'Phrase',	'Sentiment']
    writer1 = csv.DictWriter(outfile1, fieldnames = fieldnames)
    writer2 = csv.DictWriter(outfile2, fieldnames = fieldnames)
    writer3 = csv.DictWriter(outfile3, fieldnames = fieldnames)
    writer1.writeheader()
    writer2.writeheader()
    writer3.writeheader()
    for row in reader:
#        row['Phrase'] = clean(row['Phrase'])
        count = row['Phrase'].count(" ")+1
        
        if count == 1:            
            writer1.writerow(row)
        elif count == 2:
            writer2.writerow(row)
        else:
            writer3.writerow(row)
            
        
        