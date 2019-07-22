from Eng_news import formal as f1
from Eng_news2 import formal as f2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tf_idf_eng(doc_one,doc_two):
    #build corpus list of two doc
    corp=list()
    corp.append(doc_one)
    corp.append(doc_two)
    print(corp)

    vectorizer=CountVectorizer(stop_words='english') #word freq
    transformer = TfidfTransformer()
    tf_idf=transformer.fit_transform(vectorizer.fit_transform(corp))
    print(len(vectorizer.get_feature_names()))
    return tf_idf


print('The tf-idf of 2 doc are below:\n ',tf_idf_eng(f1,f2))
tfidf=tf_idf_eng(f1,f2)
arr1=tfidf.toarray()[0]
re1=arr1.reshape(1,-1)
arr2=tfidf.toarray()[1]
re2=arr2.reshape(1,-1)


def cosine_sim(firstArr,secondArr):
    return cosine_similarity(firstArr,secondArr)


print('The cosine similarity of two doc is : \n',cosine_sim(re1,re2))
