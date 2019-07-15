import requests
import re
from bs4 import BeautifulSoup
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
url='https://www.apnews.com/5fef7761f14f4c65a7de982359b87bd0'
res=requests.get(url)
soup=BeautifulSoup(res.text,'html.parser')
tmp=''
for text in soup.select('.Article p'):
    tmp+=text.get_text()
formal = re.sub('[^A-Za-z0-9]', ' ', tmp)
print(formal)
tokenized=nltk.word_tokenize(formal)
print(tokenized)
stopwords = nltk.corpus.stopwords.words('english')
content = [w for w in tokenized if w.lower() not in stopwords]
print(content)
fdist = nltk.FreqDist(content)
fdist.plot(30)