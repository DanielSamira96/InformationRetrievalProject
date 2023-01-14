from inverted_index_gcp import InvertedIndex
from BM25_from_index import BM25_from_index
from flask import Flask, request, jsonify
import gensim.downloader as api
from cosine_similarity import *
from tools import *


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def initial():
    """ Initializing the search engine by reading the inverted indexes, pageview, pagerank and word2vec, and initializing
    the necessary parameters with the ideal values.
    Returns:
     --------
        None
    """
    # initializing globals to use the data from the inverted indexes
    global idx_title
    global idx_body
    global idx_anchor
    global doc_to_title
    global pagerank
    global pageview
    global wv
    # initializing globals for the parameters of the different weights
    global weight_title
    global weight_body
    global k1_body
    global k3_body
    global b_body
    global k1_title
    global k3_title
    global b_title
    global weight_similar_query

    # setting the ideal parameters we found from the different experiments
    weight_title = 0.8
    weight_body = 1 - weight_title
    k1_body = 7.17484
    k3_body = 10.0
    b_body = 0.0
    k1_title = 0.4021
    k3_title = 4.79378
    b_title = 0.74449
    weight_similar_query = 0.03
    # reading the inverted indexes and the additional data we created, so we can use it in our search methods
    idx_title = InvertedIndex.read_index('title_index', 'title')
    idx_body = InvertedIndex.read_index('body_index', 'body')
    idx_anchor = InvertedIndex.read_index('anchor_index', 'anchor')
    doc_to_title = InvertedIndex.read_index('.', 'doc_to_title')
    pagerank = InvertedIndex.read_index('.', 'pagerank')
    pageview = InvertedIndex.read_index('.', 'pageviews-202108-user')
    # setting up the word2vec
    wv = api.load('glove-wiki-gigaword-50')


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. this method tokenizes the query in order to
    remove redundant words, then do query expansion using word2vec. with the new query, the method calculate the BM25
    algorithm for the title and for the body of the article, with the inverted index. then the method merge the
    results with ideal weights, and use the pagerank score of each document in order to add score to the search
    score.
    Returns:
    --------
         list of up to 100 search results, ordered from best to worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # tokenizing the query
    query = tokenize(query)

    # query expansion using word2vec
    query_similar = wv.most_similar(positive=query)
    query_similar_tokenize = dict([(tup[0], tup[1] * weight_similar_query) for tup in query_similar if tup[1] > 0.73])

    # computing BM25 to each inverted index by creating a BM25 objects
    bm25_title = BM25_from_index(idx_title, k1_title, k3_title, b_title)
    bm25_body = BM25_from_index(idx_body, k1_body, k3_body, b_body)
    query = Counter(query)

    # search with the BM25 algorithm
    res_bm25_title = bm25_title.search_title(query | query_similar_tokenize)
    res_bm25_body = bm25_body.search_body(query)

    # merge the results of the body and title
    res_after_merge = merge_results(res_bm25_title, res_bm25_body, weight_title, weight_body)
    # compute the score to add due to the pagerank score
    res_after_pagerank = get_results_with_pagerank(res_after_merge, pagerank)
    # return the best results in tuple (wiki_id, title) format
    res = get_titles_from_id(res_after_pagerank, doc_to_title)

    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. NO use of stemming. with the USE of the
        staff-provided tokenizer from Assignment 3 (GCP part) for the
        tokenization and remove stopwords.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # tokenize the queries with the staff-provided tokenizer
    query = Counter(tokenize(query))

    # compute the cosine similarity score
    res_cosine = get_topN_cosine_similarity_score_for_query(query, idx_body)
    # return the best results in tuple (wiki_id, title) format
    res = get_titles_from_id(res_cosine, doc_to_title)

    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # tokenize the queries with the staff-provided tokenizer
    query = Counter(tokenize(query))

    # compute the titles score for each title
    res_titles = get_ranked_boolean_for_titles(query, idx_title)

    # return the best results in tuple (wiki_id, title) format
    res = get_titles_from_id(res_titles, doc_to_title)

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # tokenize the queries with the staff-provided tokenizer
    query = Counter(tokenize(query))

    # compute the anchor score for each doc
    res_anchor = get_ranked_boolean_for_anchor(query, idx_anchor)

    # return the best results in tuple (wiki_id, title) format
    res = get_titles_from_id(res_anchor, doc_to_title)

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # returns a list of pagerank scores according to the article IDs
    res = [pagerank.get(wiki_id, 0) for wiki_id in wiki_ids]
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provided wiki articles
        had in August 2021.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # return the pageview number according to the article IDs
    res = [pageview.get(wiki_id, 0) for wiki_id in wiki_ids]
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    initial()
    app.run(host='0.0.0.0', port=8080, debug=True)
