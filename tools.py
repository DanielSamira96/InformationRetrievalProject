from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def get_topN_tf_for_titles(query, index):
    score = Counter()
    for term, posting_list in index.posting_lists_iter(query):
        for tp in posting_list:
            doc_id, freq = tp
            score[doc_id] += 1
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def get_tf_for_anchor(query, index):
    score = Counter()
    for term, posting_list in index.posting_lists_iter_anchor(query):
        for doc_id in posting_list:
            score[doc_id] += 1
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def get_titles_from_id(lst_id, doc_to_title):
    return [(item[0], doc_to_title.get(item[0], "")) for item in lst_id]


def tokenize(text):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)

    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
