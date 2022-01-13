import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

df_reviews = pd.read_csv('./naver_crawling_data/naver_cleaned_reviews.csv')
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_reviews['cleaned_reviews'])

with open('./naver_models/tfidf.pickle', 'wb') as f:
    pickle.dump(Tfidf, f)

mmwrite('./naver_models/Tfidf_book_review.mtx', Tfidf_matrix)

