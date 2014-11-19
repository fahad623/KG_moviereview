import csv


location_input_file = "../../../classifier/VW/rotten.preds.txt"
location_output_file = "../../../classifier/VW/kaggle.submission.csv"

def to_kaggle(header=""):
    print "\nReading:",location_input_file,"\nWriting:",location_output_file
    with open(location_input_file) as infile, open(location_output_file, "wb") as outfile:
        if len(header) > 0:
            outfile.write( header + "\n" )
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            outfile.write( row[1] + "," + str(int(row[0][0])-1) + "\n" )

to_kaggle("PhraseId,Sentiment")

# Kaggle score - 0.62492