

# coding: utf-8

# In[1]:


#=================================================================================================================
#Packages needed
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib import pyplot
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'qt')
import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
#=================================================================================================================
#Reducing the individual words to their respective stems
def stem(tokens):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]
    return stemmed
#=================================================================================================================
#Reduce sentences in terms of words
def tokenize(file):
    tokens=word_tokenize(file)
    words_raw=[word.lower() for word in tokens] #Converting whole text to lowercase
    words=[word for word in words_raw if word.isalpha()]
    stop_words=stopwords.words('english')
    words_nonstop=[w for w in words if not w in stop_words]
    #print(words_nonstop[:10])
    #rooted_file=stem(words_nonstop)
    library=r'C:\USERS\VARADARAYASHENOY\DESKTOP\Data\Keys.txt' #library
    #for topic-specific keywords
    key=open(library,'r')
    texts=key.read()
    key.close()
    library_keywords=word_tokenize(texts)
    lower=[word.lower() for word in library_keywords]
    nopunkt=[word for word in lower if word.isalpha()]
    stop_words1=stopwords.words('english')
    words_nonstop_key=[w for w in nopunkt if not w in stop_words1]
    lib=[w for w in nopunkt if w in words_nonstop_key]
    #rooted=stem(lib)
    diction=[w for w in words_nonstop if w in lib]
    return diction
#=================================================================================================================
#Access Directory of Corpus
def fn_CorpusFromDIR(xDIR):
    Res = dict(docs = [open(os.path.join(path,f)).read() for f in os.listdir(path)],ColNames = map(lambda x: '' + x[0:23], os.listdir(path)))
    return Res
#=================================================================================================================
#Create Term-Document Matrix
def fn_tdm_df(docs, xColNames = None, **kwargs):
    corpus=[]
    vec=CountVectorizer(decode_error='ignore',lowercase=True,analyzer='word',stop_words='english',tokenizer=tokenize)
    x1=vec.fit_transform(docs)
    #print(vocab)
    features=vec.get_feature_names()
    df = pd.DataFrame.from_records(x1.toarray().transpose(), index = features)
    df.columns = xColNames
    return vec,df,features
#=================================================================================================================
#Function to find top N words in a topic 
def show_topics(vectorizer, lda_model, n_words=20):
    docs=fn_CorpusFromDIR(path)['docs']
    vocab=vectorizer.fit(docs)
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_terms in lda_model.components_:
        top_keyword_locs = (topic_terms).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
#=================================================================================================================
#path containing files
path=r'C:\USERS\VARADARAYASHENOY\DESKTOP\Resumes' #path containing files
#================================================================================
#Creating a DTM
corpus=[]
docs = fn_CorpusFromDIR(path)['docs']
Columns=fn_CorpusFromDIR(path)['ColNames']
for dirpath,dirnames,files in os.walk(path): #opening each file in the directory
        for file in files:
            doc_corpus=[]
            filepath=dirpath + os.path.sep + file
            lines=open(filepath,errors='ignore')
            text=lines.read()
            lines.close()
            tokens=tokenize(text)
            corpus.append(tokens)
#print(corpus)
vec,d1,features= fn_tdm_df(docs,Columns,stop_words='english', charset_error = 'ignore',tokenizer=tokenize,analyzer='word')
df=d1.T
#print(df)
#======================================================================================================================
#Finding tf-idf weights
idf_trans=TfidfTransformer(norm=None)
idf=idf_trans.fit_transform(df)
idf_mat=pd.DataFrame(idf.toarray().transpose())
idf_mat.columns=fn_CorpusFromDIR(path)['ColNames']
idf_mat.index=features
IDF=idf_mat.T
svd=TruncatedSVD(n_components=2)
svd_mat=svd.fit_transform(idf)
cos_sim=cosine_similarity(svd_mat[[-1]],svd_mat)
cos_sim_df=pd.DataFrame.from_records(np.array(cos_sim))
cos_sim_df.columns=fn_CorpusFromDIR(path)['ColNames']
print(cos_sim_df)
pyplot.plot(cos_sim_df.T,'bo')
pyplot.ylabel('Cosine Similarity Value')
pyplot.show()
#====================================================================================================================
#LDA Topic Modelling
documents=fn_CorpusFromDIR(path)['ColNames']
lda_model=LatentDirichletAllocation(random_state=1,n_components=5,learning_decay=0.9,learning_method='online',learning_offset=30,max_iter=1000)
lda=lda_model.fit_transform(df)
#print(lda)
topics=["Topic:" + str(i) for i in range(lda_model.n_components)]
#print(topics)
df_doc_topic=pd.DataFrame(np.array(lda).transpose(),index=topics)
df_doc_topic.columns=fn_CorpusFromDIR(path)['ColNames']
# create a scatter plot of document-topic values
fig=pyplot.figure(num=2,figsize=(10,10))
pyplot.title('Topic 4')
pyplot.ylim(ymax=0.002)
pyplot.ylim(ymin=0.0006)
pyplot.scatter(lda[:,4],lda[:,4])
words = list(documents)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(lda[i, 4], lda[i, 4]))
pyplot.show(fig)
#fig1=pyplot.figure(num=2,figsize=(5,5))
#pyplot.plot(lda_array,topics,'+')
#print(df_doc_topic.T)
#LDA=pd.DataFrame.from_records(np.array(lda_mat).transpose(),index=topics)
#LDA.columns=fn_CorpusFromDIR(path)['ColNames']
#print(LDA)
df_topic_keys=pd.DataFrame(lda_model.components_)
df_topic_keys.columns=features
df_topic_keys.index=topics
df_topic_keys.head()
#print(df_topic_keys)
topic_keywords = show_topics(vec, lda_model, n_words=5)        
# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
top_terms=pd.DataFrame(np.array(df_topic_keywords),index=df_topic_keywords.index,columns=df_topic_keywords.columns)
top_term_table=top_terms.T
#====================================================================================================================
#Writing the DTM and TFIDF table to excel sheets
#file_path_dtm=r'C:\USERS\VARADARAYASHENOY\DESKTOP\count_table.xlsx' # Document-Term Matrix
#df.to_excel(file_path_dtm, index=corpus)
file_path_tfidf=r'C:\USERS\VARADARAYASHENOY\DESKTOP\tfidf_table.xlsx' # TF-IDF Table
IDF.to_excel(file_path_tfidf, index=features)
#file_path_doctopic=r'C:\USERS\VARADARAYASHENOY\DESKTOP\doc_topic_table.xlsx' # TF-IDF Table
#df_doc_topic.to_excel(file_path_doctopic, index=topics)
#file_path_topicword=r'C:\USERS\VARADARAYASHENOY\DESKTOP\topic_word_table.xlsx' # TF-IDF Table
#df_topic_keys.to_excel(file_path_topicword, index=features)
#file_path_topterm=r'C:\USERS\VARADARAYASHENOY\DESKTOP\top_terms_table.xlsx' # TF-IDF Table
#top_term_table.to_excel(file_path_topterm,index=topics)
#print(d1.T)
#print(idf_mat.T)
#print('\n\n')
#print(corpus)
#pyplot.show(fig)
#pyplot.show(fig1)
#=======================================================================================================================
print("\n\n")
print("--DONE--")

