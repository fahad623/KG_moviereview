# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:03:23 2014

@author: fahad
"""

from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

import pandas as pd



text = ['Hop on pop', 'Hop off pop', 'Hop Hop hop']

#vectorizer = CountVectorizer()
#
#vectorizer.fit(text)
#x = vectorizer.transform(text)
#print x.toarray()
#print vectorizer.get_feature_names()


df = pd.DataFrame({'col1': ['f', 'd', 'f', 'c'], 'col2': [11, 12, 13, 14]})
d = df.T.to_dict().values()

#vec = DictVectorizer()
#x = vec.fit_transform(d)
#print x.toarray()
#print vec.get_feature_names()


#df = pd.DataFrame({'col1': [4, 5, 6, 7], 'col2': [11, 12, 13, 14]})
#d = df.T.to_dict().values()
#vec = FeatureHasher()
#x = vec.fit_transform(d)
#print x

#text = ['Hop on the pop', 'Hop off the pop', 'Hop Hop the hop']
#vec = TfidfVectorizer()
#x = vec.fit_transform(text)
#print x.toarray()
#print vec.get_feature_names()

text = ['Hop on the pop', 'Hop off the pop', 'Hop Hop the hop']
vec = HashingVectorizer()
x = vec.fit_transform(text)
print x
