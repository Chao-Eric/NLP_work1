import requests
import re
from bs4 import BeautifulSoup
import jieba
import jieba.analyse
import nltk
url="https://newtalk.tw/news/view/2019-07-05/268891"
res=requests.get(url)
res.encoding='utf-8'
soup=BeautifulSoup(res.text)
print(soup)
tmp=''
for text in soup.select('div.fontsize.news-content p'):
    tmp+=text.get_text()
print(tmp)
formal = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', tmp)
jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('user_dict.txt')
seg_list=jieba.cut(formal,cut_all=False)
print('\n'.join(seg_list))
print("print the tfidf top 15")
tags = jieba.analyse.extract_tags(formal, 15)
for tag in tags:
    print(tag)
