import csv
import re

location_train = "../../../data/train_moreword.csv"
location_test = "../../../data/test_moreword.csv"

location_train_vw = "../../../data/rotten.train_moreword.vw" #will be created
location_test_vw = "../../../data/rotten.test_moreword.vw" #will be created

#cleans a string "I'm a string!?" returns as "i m a string"
def clean(s):
  return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()

#creates Vowpal Wabbit-formatted file from tsv file
def to_vw(location_input_file, location_output_file, test = False):
    print "\nReading:",location_input_file,"\nWriting:",location_output_file
    with open(location_input_file) as infile, open(location_output_file, "wb") as outfile:
    #create a reader to read train file
        reader = csv.DictReader(infile)
        #for every line
        for row in reader:
          #if test set label doesnt matter/or isnt available
            if test:
                label = "1"
            else:
                label = str(int(row['Sentiment'])+1)
            phrase = clean(row['Phrase'])
            outfile.write(   label + 
              " '"+row['PhraseId'] + 
              " |f " + 
              phrase + 
              " |a " + 
              "word_count:"+str(phrase.count(" ")+1)
              + "\n" )

to_vw(location_train, location_train_vw)
to_vw(location_test, location_test_vw, test=True)