from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
from scipy import stats

#def parseOutText(f):
#    content = f.split()
#    list_stemmed = []
#    for word in content:
#        if word != '\n':
#            stemmer = SnowballStemmer("english")
#            list_stemmed.append(stemmer.stem(word))

#    words = " ".join(list_stemmed)
#    return words


#if __name__ == '__main__':
#    #parseOutText("Responsive design is indemand!!")
#    b = np.array(["Responsive design is indemand!!", "Incredible hostility"])
#    newdata = np.array([parseOutText(d) for d in b])

#    print newdata


#words = " ".join([])
#test =""
#print words

df = pd.DataFrame({'a': [1.0, 4.0, 3.0], 'b':[1.0, 6.7, 9.8]})
df['mode'] = df.ix[:, 0:2].mode(axis = 1)
df.ix[df['mode'].isnull(), 'mode'] = df.ix[df['mode'].isnull(), 'a']
df['mode'] = df['mode'].astype(np.int)
print df



