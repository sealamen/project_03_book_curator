import pandas as pd
from gensim.models import Word2Vec

review_word = pd.read_csv('./naver_crawling_data/naver_cleaned_reviews.csv')

review_word.info()

cleaned_token_review = list(review_word['cleaned_reviews'])
print(cleaned_token_review[4])

cleaned_tokens = []
for sentence in cleaned_token_review:
    token = sentence.split(' ')
    cleaned_tokens.append(token)

embedding_model = Word2Vec(cleaned_tokens, size = 100,
                           window=4, min_count=20,
                           workers=4, iter=100, sg=1)

embedding_model.save('./naver_models/Word2VecModel_naver.model')
print(embedding_model.wv.vocab.keys())
print(len(embedding_model.wv.vocab.keys()))


# 단어 갯수 14995 개
# 단어 갯수 30595 개

