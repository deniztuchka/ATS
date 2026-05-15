from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 1),   # unigrams only — bigrams were creating fake phrase noise
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        stop_words=None,
    )
