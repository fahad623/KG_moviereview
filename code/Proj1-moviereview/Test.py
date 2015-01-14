from nltk.stem.snowball import SnowballStemmer
import numpy as np

def parseOutText(f):
    content = f.split()
    list_stemmed = []
    for word in content:
        if word != '\n':
            stemmer = SnowballStemmer("english")
            list_stemmed.append(stemmer.stem(word))

    words = " ".join(list_stemmed)
    return words


if __name__ == '__main__':
    #parseOutText("Responsive design is indemand!!")
    b = np.array(["Responsive design is indemand!!", "Incredible hostility"])
    newdata = np.array([parseOutText(d) for d in b])

    print newdata