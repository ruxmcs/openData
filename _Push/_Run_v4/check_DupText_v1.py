# -*- coding: utf-8 -*-

DEBUG = True
from gensim import corpora, models, similarities
import logging
from collections import defaultdict
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
"A survey of user opinion of computer system response time",
"The EPS user interface management system",
"System and human system engineering testing of EPS",
"Relation of user perceived response time to error measurement",
"The generation of random binary unordered trees",
"The intersection graph of paths in trees",
"Graph minors IV Widths of trees and well quasi ordering",
"Graph minors A survey"]

dictionary = None
tfidf = None
index = None
def Build_Tfidf(documents):
    #1.split, remove stop
    stoplist=set('for a of the and to in'.split())
    texts=[[word for word in document.lower().split() if word not in stoplist] for document in documents]
    print('-----------1----------')
    print(texts)
    #[['human', 'machine', 'interface', 'lab', 'abc', 'computer', 'applications'], ['survey', 'user', 'opinion', 'computer', 'system', 'response', 'time'],
    #['eps', 'user', 'interface', 'management', 'system'], ['system', 'human', 'system', 'engineering', 'testing', 'eps'], ['relation', 'user', 'perceived
    #', 'response', 'time', 'error', 'measurement'], ['generation', 'random', 'binary', 'unordered', 'trees'], ['intersection', 'graph', 'paths', 'trees'],
    #['graph', 'minors', 'iv', 'widths', 'trees', 'well', 'quasi', 'ordering'], ['graph', 'minors', 'survey']]

    #2. word freq
    frequency = defaultdict(int) # a dict
    # every word for freq
    for text in texts:
        for token in text:
            frequency[token]+=1
    # pick word only for freq > 1
    texts=[[token for token in text if frequency[token]>1] for text in texts]
    print('-----------2----------')
    print(texts)
    #[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], ['eps', 'user', 'interface', 'system'], ['system',
    #'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees'], ['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]

    #3.dict:  word and index
    dictionary=corpora.Dictionary(texts)
    #print(dictionary)
    #Dictionary(12 unique tokens: ['time', 'computer', 'graph', 'minors', 'trees']...)
    # dir key=word,  value=index
    print('-----------3----------')
    print(dictionary.token2id)
    #{'human': 0, 'interface': 1, 'computer': 2, 'survey': 3, 'user': 4, 'system': 5, 'response': 6, 'time': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11}

    #4.doc to Vec, using Bag of word
    new_doc = "Human computer interaction"
    # doc split using doc2bow, take word to stats using freq, word converts to index, sparse matrix to return.
    new_vec = dictionary.doc2bow(new_doc.lower().split())
    print('-----------4----------')
    print(new_vec)
    #[[(0, 1), (2, 1)]

    #5.build Corpus
    # doc to vector
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('-----------5----------')
    print(corpus)
    #[[[(0, 1), (1, 1), (2, 1)], [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(1, 1), (4, 1), (5, 1), (8, 1)], [(0, 1), (5, 2), (8, 1)], [(4, 1), (6, 1), (7, 1)], [(9, 1)], [(9, 1), (10, 1)], [(9, 1), (10, 1), (11, 1)], [(3, 1), (10, 1), (11, 1)]]

    #6 init model
    # 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
    tfidf = models.TfidfModel(corpus)

    # Test =========================
    test_doc_bow = [(0, 1), (1, 1)]
    print('-----------6----------')
    print(tfidf[test_doc_bow])
    #[(0, 0.7071067811865476), (1, 0.7071067811865476)]

    print('-----------7----------')
    #将整个语料库转为tfidf表示方法
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)

    #7.创建索引
    index = similarities.MatrixSimilarity(corpus_tfidf)

    print('-----------8----------')
    #8.相似度计算
    new_vec_tfidf=tfidf[new_vec]#将要比较文档转换为tfidf表示方法
    print(new_vec_tfidf)
    #[(0, 0.7071067811865476), (2, 0.7071067811865476)]
    print('-----------9----------')
    #计算要比较的文档与语料库中每篇文档的相似度
    sims = index[new_vec_tfidf]
    print(sims)
    #[ 0.81649655  0.31412902  0.          0.34777319  0.          0.          0.
    #  0.          0.        ]
    return dictionary, tfidf, index

def File2String(filepath):
	file= open(filepath, "r", encoding='utf8')
	#str = file.read()
	ret=""
	lines = file.readlines()
	for i in range(len(lines)):
		ret += (lines[i].rstrip('\n') + '')
	return ret

def ReadFiles(comp):
    documents = []
    dict = {}
    i = 0
    for file in comp:
        text = File2String(file)
        documents.append(text)
        dict.update({i:file})
        i = i+1
    if(DEBUG): print(documents)
    return documents, dict

def UnitTest(comp, documentx, dict, threshold = 0.8):
     ret = ""
     #img1 = cv2.imread(comp[0])
     #img2 = cv2.imread(comp[1])
     #degree = classify_gray_hist(img1,img2);print(degree)

     new_doc = File2String(comp[1])
     '''
     import copy
     documents = copy.copy(documentx)
     for i in range(len(documents)):
         if(new_doc == documents[i]):
             documents[i] = [""]
             print("Remove>", i)
    '''

     global dictionary
    #4.doc to Vec, using Bag of word
    # new_doc = "Human computer interaction"
    # doc split using doc2bow, take word to stats using freq, word converts to index, sparse matrix to return.
     new_vec = dictionary.doc2bow(new_doc.lower().split())
     print('-----------UnitTest----------', comp[1], new_doc)
     if(DEBUG): print(new_vec)
     if(DEBUG): print(dictionary, type(dictionary))

     global tfidf
     new_vec_tfidf=tfidf[new_vec]
     global index
     sims = index[new_vec_tfidf]
     print(len(sims), "SimMatrix>", sims)

     for i in range(len(sims)):
         sim = sims[i]
         if( sim >= threshold):
             #org_doc = File2String(comp[0])
             #if(DEBUG): print("i>", i, documents)
             if( len(documents[int(i)]) > 1):
                 simFile = dict[i]; print("simFile>", simFile, comp[0])
                 if( simFile == comp[0]): # only for subseq
                     ret = comp[0] +"," + comp[1]
                     if(DEBUG): print("is Sim>", ret + " | "+ str(i) + " | " + str(sim) + " | "+ str(documents[i]))
             # str(comp[0]) + "," + str(comp[1])
     #degree = classify_hist_with_split(img1,img2); print(degree)

     return ret

def StringWriteFile(urlde, filepath):
    file = open(filepath, "w")
    file.write(urlde)
    file.close()
def StringAddFile(urlde, filepath):
    file = open(filepath, "a")
    file.write(urlde)
    file.close()
import os, sys
if __name__ == '__main__':

     global dictionary
     global tfidf
     global index

     comp = []  # Prototype
     comp.append('./text_1.xml')
     comp.append('./text_2.xml')
     comp.append('./text_a.xml')
     comp.append('./text_b.xml')
     comp.append('./text_c1.xml')
     BASE_dir = "./"
     comp = []
     for file in os.listdir("./"):
         if(".xml" in file.lower()):
            comp.append(BASE_dir+file)
     documents, dict = ReadFiles(comp)  ### Build
     dictionary, tfidf, index = Build_Tfidf(documents)

     print("~~~~~~~~~ Unit-Test-1: same =====")
     comp = []
     comp.append('./text_1.xml')
     comp.append('./text_2.xml')
     # [0.5055188 1.        0.        0.        1.       ]
     if(os.path.isfile('./text_1.xml')):
         if(DEBUG): print("DEBUG file exists>", './text_1.xml')
     #new_doc = File2String(comp[1])
     res = UnitTest(comp, documents, dict, 0.5) #new_doc)
     print(res)

     print("~~~~~~~~~ Unit-Test-A: diff")
     comp.clear()
     comp.append('./text_1.xml')
     comp.append('./text_a.xml')
     if(not os.path.isfile('./text_1.xml')):
         if(DEBUG): print("DEBUG file not exists>", './text_1.xml')
     if(os.path.isfile('./text_a.xml')):
         if(DEBUG): print("DEBUG file exists>", './text_a.xml')
     UnitTest(comp, documents, dict, 0.8)
     print("~~~~~~~~~ Done with Unit-Test ==========")


     Threshold = 0.6
     if(len(sys.argv)>=2):
        Threshold = sys.argv[1]
        if(len(sys.argv)>=3):
            Debug = sys.argv[2]
            if("DEBUG" in Debug):
                DEBUG = True
            else:
                DEBUG = False

     print(">>>>>> Start Sim> ==================")
     BASE_dir = "./"
     LOG_file = "./Log_compText_v1.txt"
     StringWriteFile("", LOG_file)
     isFirst = True
     for file in os.listdir(BASE_dir):
         if(isFirst):
            filepath = BASE_dir +file
            comp.clear(); print("[Begin]<===")
            print("[0]>", filepath )
            if(".xml" in file.lower()):
                comp.append(filepath)
                isFirst = False
         else:
             filepath = BASE_dir +file
             if(".xml" in file.lower()):
                 print("[1]>", filepath)
                 comp.append( filepath )
                 ret = UnitTest(comp, documents, dict, Threshold)
                 if(len(ret)>1):
                     print("[Result on CSV]========>", ret )
                     StringAddFile(ret, LOG_file)
                     StringAddFile('\n', LOG_file)
                     print("============= ", ret )
                 else:
                     print("[Diff]")

                 comp.clear()
                 print("\n[Begin]<=================")
                 #if(".txt" in file.lower()):
                 comp.append(filepath)
                 isFirst = False

     print("END_Dup_Text")