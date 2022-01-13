import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmwrite, mmread
import pickle
from gensim.models import Word2Vec

# 데이터 가져오기
df_reviews = pd.read_csv('./naver_crawling_data/naver_cleaned_reviews.csv')
Tfidf_matrix = mmread('./naver_models/Tfidf_book_review.mtx').tocsr()
with open('./naver_models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1],
                      reverse=True)
    simScore = simScore[1:11]
    bookidx = [i[0] for i in simScore]
    recBookList = df_reviews.iloc[bookidx]
    return recBookList

embedding_model = Word2Vec.load('./naver_models/Word2VecModel_naver.model')
key_word = '감동'
sentence = [key_word] * 11
sim_word = embedding_model.wv.most_similar(key_word, topn=10)
words = []
for word, _ in sim_word:  # 앞에는 단어, 뒤에는 유사도
    words.append(word)
print(words)

for i, word in enumerate(words):
    sentence += [word] * (10-i)
sentence = ' '.join(sentence)
# print(sentence)

sentence_vec = Tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation['titles'])
