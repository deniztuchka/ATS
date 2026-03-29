from sklearn.metrics.pairwise import cosine_similarity

def cosine_score(vec_a, vec_b):
    return float(cosine_similarity(vec_a, vec_b)[0][0])
