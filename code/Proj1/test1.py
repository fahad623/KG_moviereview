# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:03:23 2014

@author: fahad
"""

from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np



text = ['Hop on pop', 'Hop off pop', 'Hop Hop hop']

#vectorizer = CountVectorizer()
#
#vectorizer.fit(text)
#x = vectorizer.transform(text)
#print x.toarray()
#print vectorizer.get_feature_names()


#df = pd.DataFrame({'col1': ['f', 'd', 'f', 'c'], 'col2': [11, 12, 13, 14]})
#d = df.T.to_dict().values()
#
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

#text = ['Hop on the pop', 'Hop off the pop', 'Hop Hop the hop']
#vec = HashingVectorizer()
#x = vec.fit_transform(text)
#print x


#text = ['not Hop on the pop', 'Hop off', 'Hop Hop hop']
#vectorizer = CountVectorizer(ngram_range=(1,8), stop_words='english')
#
#vectorizer.fit(text)
#x = vectorizer.transform(text)
#print x.toarray()
#print vectorizer.get_feature_names()


enc = OneHotEncoder()
arr = np.array([[0, 1, 2], [1,2,6]])
print arr
print arr.dtype
enc.fit(arr) 
print enc.n_values_
print enc.feature_indices_
print enc.transform([[0, 1, 6]]).toarray()