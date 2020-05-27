import os
import codecs
import json
import spacy
import pandas as pd
import itertools as it
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.matutils import cossim
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from geopy.geocoders import Nominatim
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle

import plotly.graph_objects as go

# data_directory = os.path.join('/Users/erics/Documents/Documents/UCLA/Thesis/Dataset')
conn = sqlite3.connect('../Dataset/yelp.db')
"""
overview
"""
# 59371 Restaurants from 192609 records
pd.read_sql("select count(*) from business where categories like '%Restaurants%'",conn)
#4201684 restaurant reviews from 1148098 users
sql="""
select count(*),count(distinct user_id) from review where business_id in (
    select business_id from business
    where categories like '%Restaurants%'
);
"""
pd.read_sql(sql,conn)
# 810342 restaurant tip from 252464 users
sql="""
select count(*),count(distinct user_id) from tip where business_id in (
    select business_id from business
    where categories like '%Restaurants%'
);
"""
pd.read_sql(sql,conn)


"""
LOAD DATA
"""
business = pd.read_sql("select * from business where categories like '%Restaurants%'",conn)
sql="""
select * from review 
    where business_id in (
    select business_id from business
    where categories like '%Restaurants%'
    and city = 'Las Vegas')
;
"""
review = pd.read_sql(sql,conn)
sql="""
select * from user
where user_id in (
    select user_id from review
    join business using (business_id)
    where categories like '%Restaurants%')
"""
user = pd.read_sql(sql,conn)



"""
Simple EDA
"""
#breakdown by state and city
business[['state','stars','review_count']].groupby("state").agg(['mean','count']).reset_index().sort_values([('review_count','count')],ascending=False) 
business[['address','stars','review_count']].groupby("address").agg(['mean','count']).reset_index().sort_values([('review_count','count')],ascending=False) 

#plot restaurants on map, color = rating score
#Global, US, major cities: Toronto, Las Vegas, Phoenix
"""ipyt
assign to:
refer to: https://www.bigendiandata.com/2017-06-27-Mapping_in_Jupyter/
"""

#more plots
"""
rating score open vs closed
"""
sql = """
select stars,is_open,avg(review_count) review_count,count(*) count from business where city like '%Vegas%' group by stars,is_open order by is_open,stars
"""

sql = """
select is_open,avg(review_count) review_count,count(*) count,avg(stars) from business where city like '%Vegas%' group by is_open order by is_open
"""

sql = """
select avg(review_count) review_count,count(*) count,avg(stars) from business where city like '%Vegas%' 
"""
a = pd.read_sql(sql,conn)
a[a['is_open']=='0']['count'] 

labels = ['1', '1.5', '2', '2.5', '3','3.5', '4', '4.5', '5']
open_counts = a[a['is_open']=='1']['count'].values.tolist()
# open_counts = [round(num, 1) for num in open_counts] 
closed_counts = a[a['is_open']=='0']['count'].values.tolist() 
# closed_counts = [round(num, 1) for num in closed_counts] 

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, open_counts, width, label='Open')
rects2 = ax.bar(x + width/2, closed_counts, width, label='Closed')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Restaurant Count')
ax.set_xlabel('Yelp Rating')
ax.set_title('Yelp Rating: Open vs Closed Restaurants')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


sql = """
select avg(stars),count(stars) from review where business_id in (select business_id from business where city like '%Vegas%')
"""
a = pd.read_sql(sql,conn)




















#Spacy
review = pd.read_sql('select text from review;',conn)
nlp = spacy.load('en_core_web_sm')
# parsed_review = nlp(review['text'][1:10])
for parsed_review in nlp.pipe(review['text'][1:10],
                                batch_size=10, n_threads=4):
    unigram_review = [token.lemma_ for token in parsed_review if not punct_space(token)]
    bigram_review = bigram_model[unigram_review]
    trigram_review = trigram_model[bigram_review]
    trigram_review = [term for term in trigram_review if term not in spacy.en.STOPWORDS]
    trigram_review2 = u' '.join(trigram_review)
    print(trigram_review)
    print('---------------------------')
    print(trigram_review2)
    print('===========================')

# parsed_review = nlp(review['text'])
i=0
w = []
for token in parsed_review:
    if not token.is_stop and not token.is_punct and not token.like_num and not token.is_space:
        print(token.lemma_)
        w.append(token.lemma_)
    # print(token.orth_)
    i +=1

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')

def punct_space(token):
    return token.is_punct or token.is_space

def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])


with codecs.open('unigram_sentences_all.txt', 'w', encoding='utf_8') as f:
    for sentence in lemmatized_sentence_corpus('review_text_all.txt'):
        f.write(sentence + '\n')


# abcd
"""
Focus on Restaurant in Las Vegas
"""
business_lv_sql = """
    select * from business
    where categories like '%Restaurants%'
    and city = 'Las Vegas';
"""
business_lv = pd.read_sql(business_lv_sql,conn)

user_review_lv_sql = """
    select * from review
    left join user using (user_id)
    where business_id in (select business_id from business
    where categories like '%Restaurants%'
    and city = 'Las Vegas');
"""
user_review_lv = pd.read_sql(review_lv_sql,conn)


sql = """select text,stars from review 
    where business_id in (
    select business_id from business
    where categories like '%Restaurants%'
    and city = 'Las Vegas')
;"""











#Pre-process review text data
"""
assign to:
"""








#word2vec  and A+B=C
"""
assign to:
"""


#Sentiment Analysis
#create new columns for info 
#classification model: LogisticRegression, LinearSVC, SGDClassifier, RandomForest
#output all model to pickle file
"""
assign to: Eric
"""
l = []
r = review['stars']
for a in review['text']:
    # a = review['text'][i]
    l.append(TextBlob(a).sentiment[0])
    # if TextBlob(a).sentiment[0]< 0.05:
    #     print(a)
    #     print(TextBlob(a).sentiment)
np.corrcoef(r,l)
review_txt_filepath = os.path.join('review_text_all.txt')
review_count = 0




# given an address and type of food, recommend:
# get location for given address
address = "3655 S Las Vegas Blvd, Las Vegas, NV 89109"
geolocator = Nominatim(user_agent="stat418_project")
location = geolocator.geocode(address) 
location.longitude
location.latitude
#filter restaurants close by 
# rate those restaurants by similarity
# look at user's review, how much does he focus on the 5 subtopics
# look at the type of restaurants this user goes to
# find restaurants have high LDA Document Similarity




#stalking user:
#user's path base on the restaurants review
#user's sentiment toward food and restaurant features
"""
assign to:
"""



#APIs
# load all pre-saved model to make predictions\
"""
assign to:
"""



#UI
#design



sql="""
select * from tip where business_id in (
    select business_id from business
    where categories like '%Restaurants%'
);
"""
pd.read_sql(sql,conn)




####wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
sql="""
select * from parsed_review 
where business_id in (select business_id from business where city like '%Vegas%' and stars >= 4)
and review_id in (select review_id from review where stars > 4);
"""
a = pd.read_sql(sql,conn)

good_review = ' '.join(a['clean_review'].values)

wordcloud = WordCloud().generate(good_review)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
sql="""
select * from parsed_review 
where business_id in (select business_id from business where city like '%Vegas%' and stars > 4)
and review_id in (select review_id from review where stars > 4);
"""
a = pd.read_sql(sql,conn)

good_review = ' '.join(a['clean_review'].values)

wordcloud = WordCloud(background_color="white").generate(good_review)
plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')
plt.axis("off")
plt.show()