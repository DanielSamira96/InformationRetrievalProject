from collections import Counter
from nltk.corpus import stopwords
import nltk
import math
import re
nltk.download('stopwords')


def get_ranked_boolean_for_titles(query, index):
    ''' Returns ALL search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title.
    Returns:
    --------
        list of ALL search results, ordered from best to
        worst where each element is a tuple (wiki_id, score).
    '''
    score = Counter()
    for term, posting_list in index.posting_lists_iter(query):
        for tp in posting_list:
            doc_id, freq = tp
            score[doc_id] += 1
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def get_ranked_boolean_for_anchor(query, index):
    ''' Returns ALL  search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
    Returns:
    --------
        list of ALL search results, ordered from best to
        worst where each element is a tuple (wiki_id, score).
    '''
    score = Counter()
    for term, posting_list in index.posting_lists_iter_anchor(query):
        for doc_id in posting_list:
            score[doc_id] += 1
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def get_titles_from_id(lst_id, doc_to_title):
    ''' Returns a list of tuples with doc_id and the title of each document in the given IDs list
    Returns:
    --------
        list of ALL search results, where each element is a tuple (wiki_id, title).
    '''
    return [(item[0], doc_to_title.get(item[0], "")) for item in lst_id]


def tokenize(text):
    '''
    tokenizing the received string text with the use of the staff-provided tokenizer from Assignment 3 (GCP part) for
    the tokenization and remove stopwords.
    Returns:
    --------
     list of tokens : list of strings
    '''
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)

    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]


def get_results_with_pagerank(lst_id, pagerank):
    """ Returns scores values for a list of provided wiki article IDs, considering the pagerank score of the documents.

    Returns:
    --------
        list of tuples: list of calculated scores that correspond to the provided article IDs.
    """
    return sorted([(item[0], max(math.log(pagerank.get(item[0], 100), 100), 1) * item[1]) for item in lst_id], key=lambda x: x[1], reverse=True)


def merge_results(title_scores, body_scores, weight_title=0.5, weight_body=0.5, N=100):
    """
    This function merge and sort documents retrieved by its weight score.
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: score
    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: score
    title_weight: float, for weighted average utilizing title and body scores
    text_weight: float, for weighted average utilizing title and body scores
    N: Integer. How many document to retrieve. By default, N = 100.
    Returns:
    -----------
    list of tuples of the topN as follows: (doc_id, calculated score)
    """
    data_ans = {}
    for docId, docScore in title_scores.items():
        data_ans[docId] = docScore * weight_title
    for docId, docScore in body_scores.items():
        data_ans[docId] = data_ans.get(docId, 0) + docScore * weight_body
    return sorted(data_ans.items(), key=lambda x: x[1], reverse=True)[:N]
