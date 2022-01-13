import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QStringListModel
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
import bookqt_rc

form_window = uic.loadUiType('./book_image/bookcurator3.ui')[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.df_reviews = pd.read_csv('./naver_crawling_data/naver_cleaned_reviews.csv')
        self.Tfidf_matrix = mmread('./naver_models/Tfidf_book_review.mtx').tocsr()
        self.embedding_model = Word2Vec.load('./naver_models/Word2VecModel_naver.model')
        with open('./naver_models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.titles = list(self.df_reviews['titles'])
        self.titles.sort()
        for title in self.titles:
            self.cmb_titles.addItem(title)
        # QCompleter 는 일반적으로 QLineEdit 또는 QComboBox 와 함께 사용
        model = QStringListModel()
        model.setStringList(self.titles)

        completer = QCompleter()
        completer.setModel(model)

        self.le_keyword.setCompleter(completer)
        self.cmb_titles.currentIndexChanged.connect(self.cmb_titles_slot)
        self.btn_recommend.clicked.connect(self.btn_recommend_slot)

    def cmb_titles_slot(self):
        title = self.cmb_titles.currentText()
        recommendation_titles = self.recommend_by_book_title(title)
        self.lbl_result1.setText(recommendation_titles[0])
        self.lbl_result2.setText(recommendation_titles[1])
        self.lbl_result3.setText(recommendation_titles[2])
        self.lbl_result4.setText(recommendation_titles[3])
        self.lbl_result5.setText(recommendation_titles[4])

    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1],
                          reverse=True)
        simScore = simScore[1:11]
        bookidx = [i[0] for i in simScore]
        recBookList = self.df_reviews.iloc[bookidx]
        return recBookList['titles']

    def btn_recommend_slot(self):
        key_word = self.le_keyword.text()
        if key_word:
            if key_word in self.titles:
                recommendation_titles = self.recommend_by_book_title(key_word)
                self.lbl_result1.setText(recommendation_titles[0])
                self.lbl_result2.setText(recommendation_titles[1])
                self.lbl_result3.setText(recommendation_titles[2])
                self.lbl_result4.setText(recommendation_titles[3])
                self.lbl_result5.setText(recommendation_titles[4])
            else:
                key_word = key_word.split()
                if len(key_word) > 20:
                    key_word = key_word[:20]
                if len(key_word) > 10:
                    sentence = ' '.join(key_word)
                    recommendation_titles = self.recommend_by_sentence(sentence)
                    self.lbl_result1.setText(recommendation_titles[0])
                    # self.lbl_recommend2.setText(recommendation_titles[1])
                    # self.lbl_recommend3.setText(recommendation_titles[2])
                    # self.lbl_recommend4.setText(recommendation_titles[3])
                    # self.lbl_recommend5.setText(recommendation_titles[4])
                else:
                    sentence = [key_word[0]] * 11
                    try:
                        sim_word = self.embedding_model.wv.most_similar(key_word[0], topn=10)

                    except:
                        self.lbl_result1.setText('제가 모르는 단어에요 ㅠㅠ')
                        self.lbl_result2.setText('미안해요.=>.<=')
                        self.lbl_result3.setText('다른 키워드를 입력해 주세요')
                        self.lbl_result4.setText('')
                        self.lbl_result5.setText('')
                        return
                    words = []
                    for word, _ in sim_word:
                        words.append(word)
                    for i, word in enumerate(words):
                        sentence += [word] * (10 - i)
                    sentence = ' '.join(sentence)

                    recommendation_titles = self.recommend_by_sentence(sentence)
                    self.lbl_result1.setText(recommendation_titles[0])
                    self.lbl_result2.setText(recommendation_titles[1])
                    self.lbl_result3.setText(recommendation_titles[2])
                    self.lbl_result4.setText(recommendation_titles[3])
                    self.lbl_result5.setText(recommendation_titles[4])

    def recommend_by_sentence(self, sentence):
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
        recommendation_titles = self.getRecommendation(cosine_sim)
        recommendation_titles = list(recommendation_titles)
        return recommendation_titles

    def recommend_by_book_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation_titles = self.getRecommendation(cosine_sim)
        recommendation_titles = list(recommendation_titles)
        return recommendation_titles


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())
