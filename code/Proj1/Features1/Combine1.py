import numpy as np
import pandas as pd
import shutil
import os

inputFile1 = "../../../data/kaggle.submission_oneword.csv"
inputFile2 = "../../../data/kaggle.submission_moreword.csv"
clfFolder = "../../../classifier/VW_combine1/"

   
    
if __name__ == '__main__':
    
    shutil.rmtree(clfFolder, ignore_errors=True)

    df_input1 = pd.read_csv(inputFile1)
    df_input2 = pd.read_csv(inputFile2)
    
    df_output = pd.concat([df_input1, df_input2])
    df_output = df_output.sort(columns = ['PhraseId'])
    
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)
    df_output.to_csv(clfFolder + "output.csv", index = False)
    
