import math


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
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
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
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
        # YOUR CODE HERE
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


def merge_results(title_scores, body_scores, weight_title=0.5, weight_body=0.5, N=100):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE
    data_ans = {}
    for docId, docScore in title_scores.items():
        data_ans[docId] = docScore * weight_title
    for docId, docScore in body_scores.items():
        data_ans[docId] = data_ans.get(docId, 0) + docScore * weight_body
    return sorted(data_ans.items(), key=lambda x: x[1], reverse=True)[:N]
