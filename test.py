import requests
import re
from bs4 import BeautifulSoup
import jieba
import jieba.analyse
import nltk
res=requests.get("https://tw.news.yahoo.com/%E6%9F%AF%E6%96%87%E5%93%B2%E5%8A%89%E7%B5%90-%E5%B0%87%E7%A2%B0%E9%9D%A2-%E8%94%A1%E8%8B%B1%E6%96%87%E5%97%86-%E5%85%A9%E5%B2%B8%E4%BA%8B%E5%8B%99%E6%98%AF%E4%B8%AD%E5%A4%AE%E8%81%B7%E6%AC%8A-070242321.html")
soup=BeautifulSoup(res.text,"html.parser")
tmp=''
for text in soup.select('p'):
    tmp+=text.get_text()

formal = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', tmp)
jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('user_dict.txt')
seg_list=jieba.cut(formal,cut_all=False)
print('\n'.join(seg_list))
print("print the tfidf top 15")
tags = jieba.analyse.extract_tags(formal, 15)
for tag in tags:
    print(tag)
