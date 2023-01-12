from flask import Flask, request, jsonify
from inverted_index_gcp import InvertedIndex
from BM25_from_index import *
from cosine_similarity import *
from tools import *


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# # config
# k1 = 1.5
# k3 = 2
# b = 0.75
# weight_title = 0.5
# weight_body = 1 - weight_title


def initial():
    global idx_title
    global idx_body
    global idx_anchor
    global doc_to_title
    global pagerank
    global pageview
    global weight_title
    global weight_body
    global k1
    global k3
    global b
    weight_title = 0.5
    weight_body = 1 - weight_title
    k1 = 1.5
    k3 = 2
    b = 0.75
    idx_title = InvertedIndex.read_index('title_index', 'title')
    idx_body = InvertedIndex.read_index('body_index', 'body')
    idx_anchor = InvertedIndex.read_index('anchor_index', 'anchor')
    doc_to_title = InvertedIndex.read_index('.', 'doc_to_title')
    pagerank = InvertedIndex.read_index('.', 'pagerank')
    pageview = InvertedIndex.read_index('.', 'pageviews-202108-user')


@app.route("/params", methods=['POST'])
def params():
    global weight_title
    global weight_body
    global k1
    global k3
    global b
    res = []
    params = request.get_json()

    if len(params) == 0:
        return jsonify(res)

    weight_title = params[0]
    weight_body = 1 - weight_title
    k1 = params[1]
    k3 = params[2]
    b = params[3]

    return jsonify(res)


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)

    # read posting lists from disk
    bm25_title = BM25_from_index(idx_title, k1, k3, b)
    # booleanTitle = get_topN_tf_for_titles(query, idx_title)
    bm25_body = BM25_from_index(idx_body, k1, k3, b)
    # res = merge_results(dict(booleanTitle), bm25_body.search_body(query), weight_title, weight_body)
    res = merge_results(bm25_title.search_title(query), bm25_body.search_body(query), weight_title, weight_body)


    # cosine = get_topN_cosine_similarity_score_for_query(query, idx_body)
    # res = merge_results(bm25_title.search_title(query), dict(cosine), weight_title, weight_body)

    res = get_titles_from_id(res, doc_to_title)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    query = tokenize(query)

    res = get_topN_cosine_similarity_score_for_query(query, idx_body)
    res = get_titles_from_id(res, doc_to_title)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)

    res = get_topN_tf_for_titles(query, idx_title)
    res = get_titles_from_id(res, doc_to_title)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    query = tokenize(query)
    res = get_tf_for_anchor(query, idx_anchor)
    res = get_titles_from_id(res, doc_to_title)

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [pagerank.get(wiki_id, 0) for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [pageview.get(wiki_id, 0) for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)




if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    initial()
    app.run(host='0.0.0.0', port=8080, debug=True)
