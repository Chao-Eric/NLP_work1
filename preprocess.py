import requests
import re
import jieba
import jieba.analyse
from bs4 import BeautifulSoup
import nltk

def get_cont(url):
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, "html.parser")
    return soup


def formalized_ch(text):
    return re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)


def formalized_eng(text):
    return re.sub('[^A-Za-z0-9]', ' ', text)


def cut_word_ch(formalText):
    jieba.set_dictionary('dict.txt.big')
    jieba.load_userdict('user_dict.txt')
    return jieba.cut(formalText, cut_all=False)


def extract_stop_word(tokenizedWord):
    stopwords = nltk.corpus.stopwords.words('english')
    return [w for w in tokenizedWord if w.lower() not in stopwords]


def tf_idf_ch(formal,seg_list):
    return jieba.analyse.extract_tags(formal, topK=len(list(seg_list)),withWeight=True)



