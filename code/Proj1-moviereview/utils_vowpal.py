import csv
import re
import pre_process

submission_output_file = "../../data/kaggle.submission.csv"

#cleans a string "I'm a string!?" returns as "i m a string"
def clean(s):
  return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()

#creates Vowpal Wabbit-formatted file from tsv file
def to_vw(location_input_file, location_output_file, test = False):
    print "\nReading:",location_input_file,"\nWriting:",location_output_file
    with open(location_input_file) as infile, open(location_output_file, "wb") as outfile:
    #create a reader to read train file
        reader = csv.DictReader(infile, delimiter="\t")
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

def to_kaggle(location_input_file, location_output_file, header=""):
    print "\nReading:",location_input_file,"\nWriting:",location_output_file
    with open(location_input_file) as infile, open(location_output_file, "wb") as outfile:
        if len(header) > 0:
            outfile.write( header + "\n" )
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            outfile.write( row[1] + "," + str(int(row[0][0])-1) + "\n" )

if __name__ == '__main__':
#    to_vw(pre_process.trainFile, pre_process.trainFile_vw)
#    to_vw(pre_process.testFile, pre_process.testFile_vw, test=True)

    to_kaggle(pre_process.predsFile_vw,  submission_output_file, "PhraseId,Sentiment")