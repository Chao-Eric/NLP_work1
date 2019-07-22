from Chi_news import formal as f1
from Chi_news2 import formal as f2
import preprocess as pre
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tf_idf_chi(doc_one,doc_two):
    #build corpus list of two doc
    corp=list()
    corp.append(doc_one)
    corp.append(doc_two)
    print(corp)

    vectorizer=CountVectorizer() #word freq
    transformer = TfidfTransformer()
    tf_idf=transformer.fit_transform(vectorizer.fit_transform(corp))
    print(len(vectorizer.get_feature_names()))
    return tf_idf


def cosine_sim(firstArr,secondArr):
    return cosine_similarity(firstArr,secondArr)


seg_list_one=pre.cut_word_ch(f1)
seg_list_two=pre.cut_word_ch(f2)

tfidf=tf_idf_chi(' '.join(seg_list_one),' '.join(seg_list_two))
arr1=tfidf.toarray()[0]
re1=arr1.reshape(1,-1)
arr2=tfidf.toarray()[1]
re2=arr2.reshape(1,-1)

print('The cosine similarity of two doc is : \n',cosine_sim(re1,re2))
