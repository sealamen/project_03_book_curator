import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
from konlpy.tag import Okt
from matplotlib import font_manager, rc
import matplotlib as mpl
import numpy as np

# 폰트 적용
font_path = './malgun.ttf'
font_name = font_manager.FontProperties(
    fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)

df = pd.read_csv('./naver_crawling_data/naver_cleaned_reviews.csv')
print(df.head())


words = df.iloc[0,1]
words = words.split()
print(words)

worddict = collections.Counter(words)
worddict = dict(worddict)
print(worddict)

# wordcloud 이미지 만들기
wordcloud_img = WordCloud(
    background_color='white', max_words=2000,
    font_path=font_path).generate_from_frequencies(worddict)

# wordcloud 이미지 그리기

plt.figure(figsize=(12,12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()

# stopwords 빼고 그리기
stopwords = []

from PIL import Image

book_mask = np.array(Image.open('./book_image_3.png'))

wordcloud_img = WordCloud(
    background_color='white', max_words=2000,
    font_path=font_path, collocations=False, mask=book_mask,
    stopwords=stopwords).generate(df.cleaned_reviews[0])

plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()