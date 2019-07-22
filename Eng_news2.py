import re
import nltk
import preprocess as pre
url='https://www.concordmonitor.com/Trump-ask-Americans-to-stay-true-to-our-cause-26770267'
soup=pre.get_cont(url)
tmp=''
for text in soup.select('#articlebody p'):
    tmp+=text.get_text()
formal = pre.formalized_eng(tmp)
print(formal)
tokenized=nltk.word_tokenize(formal)
print(tokenized)
content = pre.extract_stop_word(tokenized)
print(content)
fdist = nltk.FreqDist(content)
fdist.plot(30)