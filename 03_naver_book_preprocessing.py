

# 기존의 파일(yes24_cleaned_data)에서

# 3. stopwords 및 눈에 보이는 몇 단어(stopwords_book) 추가로 제거
# 4. okt를 통해 형태소 분리하고 okt.pos에서 명사, 동사, 형용사, 부사를 살림
# 5. 이를 통해 발생하는 빈셀 제거 후 reset_index(4867 > 4867)(다 잘 차있음)


import pandas as pd
from konlpy.tag import Okt
import re
import numpy as np


df = pd.read_csv('./naver_crawling_data/naver_book_reviews_onesentence.csv')
print(df)
print(df.shape)

okt = Okt()

stopwords = pd.read_csv('./stopwords.csv', index_col=0)
# print(stopwords)

stopwords_list = list(stopwords['stopword'])
print(stopwords_list)
print(type(stopwords))
print(type(stopwords_list))

stopwords_book = ['책소개', '소설', '소개', '문학평론가', '평론가', '문학평론가',
                  '작가', '주인공', '작가', '저자', '작품', '이야기', '문학',
                  '소설가', '출판', '창작']

count = 0
cleaned_sentences = []
for sentence in df.reviews:
    count += 1
    if count % 10 == 0:
        print('.', end='')
    if count % 100 == 0:
        print()
    sentence = re.sub('[^가-힣 ]', '', sentence)
    token = okt.pos(sentence, stem=True)
    df_token = pd.DataFrame(token, columns = ['word', 'class'])
    df_cleaned_token = df_token[(df_token['class'] == 'Noun') |
                                (df_token['class'] == 'Verb') |
                                (df_token['class'] == 'Adjective') |
                                (df_token['class'] == 'Adverb')]
    words = []
    for word in df_cleaned_token['word']:
        if len(word) > 1:
            if word not in stopwords_book:
                if word not in stopwords_list:
                    words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
df['cleaned_reviews'] = cleaned_sentences
print(df.head())
df.info()
df = df[['titles', 'cleaned_reviews']]
# df.dropna(inplace=True)
# df.reset_index(drop=True, inplace=True)
# 왜 안되지? A: null로 인식하지 못하는 경우가 있으므로 replace로 nan값으로 바꿔서 제거
df['cleaned_reviews'].replace('', np.nan, inplace=True)
df.dropna(subset=['cleaned_reviews'], inplace=True)
df.reset_index(drop=True, inplace=True)  # drop=True를 줄 경우 기존 인덱스 버림
df.info()
df.to_csv('./naver_crawling_data/naver_cleaned_reviews.csv', index=False)

