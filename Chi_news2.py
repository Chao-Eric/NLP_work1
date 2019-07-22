import jieba.analyse
import preprocess as pre
url="https://newtalk.tw/news/view/2019-07-05/268891"
soup=pre.get_cont(url)
tmp=''
for text in soup.select('div.fontsize.news-content p'):
    tmp+=text.get_text()
print(tmp)

formal = pre.formalized_ch(tmp)
seg_list_two=pre.cut_word_ch(formal)

print("print the tfidf")
tags = pre.tf_idf_ch(formal,seg_list_two)
print(tags)
