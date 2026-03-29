from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,        # iki dokümanda ortak geçen terimleri eleme
        sublinear_tf=True,
        stop_words=None     # istersen "english" yapabilirsin; kritik olan max_df
    )
