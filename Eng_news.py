import nltk
import preprocess as pre

url='https://www.apnews.com/5fef7761f14f4c65a7de982359b87bd0'
soup=pre.get_cont(url)
tmp=''
for text in soup.select('.Article p'):
    tmp+=text.get_text()
formal = pre.formalized_eng(tmp)
print(formal)
tokenized=nltk.word_tokenize(formal)
print(tokenized)

content = pre.extract_stop_word(tokenized)
print(content)
fdist = nltk.FreqDist(content)
fdist.plot(30)
print('td-idf: \n')
