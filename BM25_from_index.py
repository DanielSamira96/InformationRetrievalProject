import math


class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    k3 : float, default 2
    b : float, default 0.75
    index : inverted index
    N : amount of documents in the index
    AVGDL : average words in a document
    """
    def __init__(self, index, k1=1.5, k3=2, b=0.75):
        self.b = b
        self.k1 = k1
        self.k3 = k3
        self.index = index
        self.N = index.N
        self.AVGDL = index.avgdl

    def search_title(self, query):
        """
        This function calculate the bm25 score on the titles of the documents for given query.
        We only check documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                key: doc_id
                                                                value: BM25 score
        Parameters:
        -----------
        query: dictionary of tokens representing the query, and it's tf.
        Returns:
        -----------
        score: dictionary, BM25 score.
        """
        score = {}
        for term, posting_list in self.index.posting_lists_iter(query):
            idf = math.log((1 + self.N) / self.index.df[term], 10)
            tfq = ((self.k3 + 1) * query[term]) / (self.k3 + query[term])
            for tp in posting_list:
                doc_id, freq = tp
                doc_len = self.index.DL[doc_id]
                numerator = idf * freq * (self.k1 + 1) * tfq
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.AVGDL))
                score[doc_id] = score.get(doc_id, 0.0) + (numerator / denominator)
        return score

    def search_body(self, query):
        """
        This function calculate the bm25 score on the bodies of the documents for given query.
        We only check documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                key: doc_id
                                                                value: BM25 score
        Parameters:
        -----------
        query: dictionary of tokens representing the query, and it's tf.
        Returns:
        -----------
        score: dictionary, BM25 score.
        """
        score = {}
        for term, posting_list in self.index.posting_lists_iter(query):
            idf = math.log((1 + self.N) / self.index.df[term], 10)
            tfq = ((self.k3 + 1) * query[term]) / (self.k3 + query[term])
            for tp in posting_list:
                doc_id, freq = tp
                doc_len = self.index.DL[doc_id][0]
                numerator = idf * freq * (self.k1 + 1) * tfq
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score[doc_id] = score.get(doc_id, 0.0) + (numerator / denominator)
        return score
