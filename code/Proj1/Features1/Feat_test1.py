import csv
import re

location_test = "../../../data/test.tsv"

location_test_oneword = "../../../data/test_oneword.csv"

location_test_moreword = "../../../data/test_moreword.csv"


def clean(s):
    return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
  
  

with open(location_test) as infile, open(location_test_oneword, 'wb') as outfile1,  open(location_test_moreword, 'wb') as outfile2:
    reader = csv.DictReader(infile, delimiter="\t")
    fieldnames = ['PhraseId',	'SentenceId',	'Phrase']
    writer1 = csv.DictWriter(outfile1, fieldnames = fieldnames)
    writer2 = csv.DictWriter(outfile2, fieldnames = fieldnames)
    writer1.writeheader()
    writer2.writeheader()
    for row in reader:
#        row['Phrase'] = clean(row['Phrase'])
        count = row['Phrase'].count(" ")+1
        
        if count == 1:            
            writer1.writerow(row)        
        else:
            writer2.writerow(row)
            
        
        