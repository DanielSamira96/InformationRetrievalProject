import math


def get_topN_cosine_similarity_score_for_query(query, index, N=100):
    score = {}
    for term, posting_list in index.posting_lists_iter(query):
        idf = math.log((index.N + 1) / index.df[term], 10)
        for tp in posting_list:
            doc_id, tf = tp
            score[doc_id] = score.get(doc_id, 0.0) + (query[term] * tf * idf)
    norm_query = math.sqrt(sum([math.pow(tf, 2) for tf in query.values()]))
    for doc_id, val in score.items():
        dl = index.DL[doc_id][1]
        if dl == 0:
            dl = 0.01
        score[doc_id] = val * (1 / norm_query) * dl
    return sorted(score.items(), key=lambda x: x[1], reverse=True)[:N]
