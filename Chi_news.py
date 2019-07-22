import jieba.analyse
import preprocess as pre

url="https://tw.news.yahoo.com/%E6%9F%AF%E6%96%87%E5%93%B2%E5%8A%89%E7%B5%90-%E5%B0%87%E7%A2%B0%E9%9D%A2-%E8%94%A1%E8%8B%B1%E6%96%87%E5%97%86-%E5%85%A9%E5%B2%B8%E4%BA%8B%E5%8B%99%E6%98%AF%E4%B8%AD%E5%A4%AE%E8%81%B7%E6%AC%8A-070242321.html"
soup=pre.get_cont(url)

#get pure text
tmp=''
for text in soup.select('p'):
    tmp+=text.get_text()

formal = pre.formalized_ch(tmp)
#define dictionary of jieba

seg_list_one=pre.cut_word_ch(formal)


#test print the result of cutting

print("print the tfidf")
tags = pre.tf_idf_ch(formal,seg_list_one)
for tfidf in tags:
    print(tfidf[1])

#
#    jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
#    sentence   為待提取的文本
#    topK    為返回幾個TF / IDF
#    權重最大的關鍵詞，默認值為20
#    withWeight    為是否一併返回關鍵詞權重值，默認值為False
#    allowPOS    僅包括指定詞性的詞，默認值為空，即不篩選
#    jieba.analyse.TFIDF(idf_path=None)
#    新建TFIDF實例，idf_path為IDF頻率文件

