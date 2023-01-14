# Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from tools import *
from sklearn.model_selection import KFold  # For K-fold cross validation

nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import json
# from mealpy.swarm_based.GWO import BaseGWO
from mealpy.math_based.HC import OriginalHC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcloros
import requests
from time import time
import math
from datetime import datetime
import os


# def tokenize(text):
#     RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
#     english_stopwords = frozenset(stopwords.words('english'))
#     corpus_stopwords = ["category", "references", "also", "external", "links",
#                         "may", "first", "see", "history", "people", "one", "two",
#                         "part", "thumb", "including", "second", "following",
#                         "many", "however", "would", "became"]
#
#     all_stopwords = english_stopwords.union(corpus_stopwords)
#
#     return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
#
#
# ## Exapmple document (list of sentences)
# doc = ["I love data science",
#        "I love coding in python",
#        "I love building NLP tool",
#        "This is a good phone",
#        "This is a good TV",
#        "This is a good laptop"]
#
# # Tokenization of each document
# tokenized_doc = []
# for d in doc:
#     tokenized_doc.append(tokenize(d.lower()))
# print(tokenized_doc)
#
# # Convert tokenized document into gensim formated tagged data
# tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
# print(tagged_data)
#
# ## Train doc2vec model
# model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs=100)
# # Save trained doc2vec model
# model.save("test_doc2vec.model")
# ## Load saved doc2vec model
# model = Doc2Vec.load("test_doc2vec.model")
# ## Print model vocabulary
# print(model.wv.index_to_key)
#
# # find most similar doc
# test_doc = tokenize("That is a good device".lower())
# print(model.dv.most_similar(positive=[model.infer_vector(test_doc)], topn=5))


# def doc2Vec():
#     ## Load saved doc2vec model
#     model = Doc2Vec.load("test_doc2vec.model")


# model = Doc2Vec.load("doc2vec_wiki_d300_n5_w8_mc50_t12_e10_dbow.model")



# url = 'http://34.122.57.217:8080'
#
# res = requests.get(url + '/search', {'query': q}, timeout=35)


# "wiki-en"
# wv = api.load('wiki-en')
# wv = api.load('glove-wiki-gigaword-50')
# # original query
# query = tokenize('best marvel movie')
# print(query)
# print(wv.most_similar(positive=query, topn=10))
# print(wv.most_similar(positive=["best"], topn=10))
# print(wv.most_similar(positive=["marvel"], topn=10))
# print(wv.most_similar(positive=["movie"], topn=10))
#
# exit()

# with open('queries_train.json', 'rt') as f:
#     queries = json.load(f)


# model.fit(data[predictors], data[outcome])
# return accuracy1, np.mean(cross_validation_score)


import json
# from mealpy.swarm_based.GWO import BaseGWO
from mealpy.math_based.HC import OriginalHC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcloros
import requests
from time import time
import math
from datetime import datetime
import os

with open('queries_train.json', 'rt') as f:
    queries = json.load(f)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def recall(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


url = 'http://34.122.57.217:8080'

global index
global max_value
global d_max
global x_axis
global y_axis
global interval
global startTime
global current_queries
index = 1
x_axis = []
y_axis = []
epoch = 1
max_value = 0
d_max = []
interval = 15
startTime = datetime.now().strftime("%H_%M_%S")


def calculateMap(d):
    global index
    global max_value
    global d_max
    global x_axis
    global y_axis
    global interval
    d = list(d)
    requests.post(url + '/params', json=d)
    av_total = 0
    for i in current_queries:
        q, true_wids = queries[i]
        try:
            res = requests.get(url + '/search', {'query': q}, timeout=35)
            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                av_total += average_precision(true_wids, pred_wids)
        except:
            pass
    result = av_total / len(current_queries)
    print("params:" + str(d) + ", map:" + str(result))
    x_axis.append(index)
    y_axis.append(result)
    if result > max_value:
        max_value = result
        d_max = d
    if index % interval == 0:
        # Plot lists 'x' and 'y'
        plt.plot(x_axis, y_axis)

        # Plot axes labels and show the plot
        plt.xlabel('Iteration Index')
        plt.ylabel('Map@40')
        plt.xticks(range(0, index, math.ceil(index / 10.0)))
        # plt.show()
        if not os.path.exists(f'data/{startTime}'):
            os.mkdir(f'data/{startTime}')
        plt.savefig(f'data/{startTime}/{str(index)}.png')
        print("Solution: [" + str(d_max) + "], globalMax: " + str(max_value))
    index += 1
    return result


def generate_position(lb=None, ub=None):
    """Start from a non-random position"""
    return np.array([0.5, 1.5, 2, 0.75])


problem_dict1 = {
    "fit_func": calculateMap,
    "lb": [0.3, 0, 0, 0, 0, 0, 0, 0.1],
    "ub": [0.8, 10, 10, 1, 10, 10, 1, 0.7],
    "minmax": "max"
    # "generate_position": generate_position
}

pop_size = 10
neighbour_size = 5

queries = list(queries.items())
kf = KFold(n_splits=5)
cross_validation_score = []
model = OriginalHC(epoch, pop_size, neighbour_size)
for train, test in kf.split(queries):
    current_queries = list(train)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, mapTrain: {best_fitness}")

    d = list(best_position)
    requests.post(url + '/params', json=d)
    av_total = 0
    for i in list(test):
        q, true_wids = queries[i]
        try:
            res = requests.get(url + '/search', {'query': q}, timeout=35)
            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                av_total += average_precision(true_wids, pred_wids)
        except:
            pass
    result = av_total / len(list(test))
    print("params:" + str(d) + ", mapTest:" + str(result))
    print("params:" + str(d) + ", avgMapTrainTest:" + str((result + best_position) / 2))













# # dataset = api.load("text8")  # load dataset as iterable
# model = Word2Vec(wv)  # train w2v model

# # find synonyms of the query
# synonyms = []
# for syn in wv.most_similar(positive=query, topn=10):
#     for lemma in syn.lemmas():
#         synonyms.append(lemma.name())
#
# # add synonyms to the query
# expanded_query = query + " " + " ".join(synonyms)
# print(expanded_query)


