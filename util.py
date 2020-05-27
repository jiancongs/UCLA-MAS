import os
import codecs
import json
import spacy
import pandas as pd
import itertools as it
import sqlite3
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from geopy.geocoders import Nominatim
from textblob import TextBlob
from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser

conn = sqlite3.connect('../Dataset/yelp.db')

def get_LV_restaurants(limit=1000000):
    sql="""
        select * from business 
            where categories like '%Restaurants%' 
            and city like '%Las Vegas%' 
            limit = {}
        """.format(limit)
    return pd.read_sql(sql,conn)

def get_LV_reviews(limit=100000000):
    sql="""
        select * from review 
            where business_id in (
            select business_id from business
            where categories like '%Restaurants%'
            and city = 'Las Vegas')
            limit = {};
        """.format(limit)
    return pd.read_sql(sql,conn)

def get_parsed_review(limit=100000000):
    sql="selct * from parsed_review limit = {}".format(limit)
    return pd.read_sql(sql,conn)


def run_sql(sql):
    with sqlite3.connect('../Dataset/yelp.db') as con:
        c = con.cursor()
        c.execute(sql)
        con.commit()

"""
Below are data prep functions for Unigram, Bigram, Trigram
one time job to generate the files and then save the models
"""
def punct_space(token):
    return token.is_punct or token.is_space

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')
            
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

def review_json_txt():
    with codecs.open('../Dataset/review_text_all.txt', 'w', encoding='utf_8') as review_txt_file:
        # open the existing review json file
        with codecs.open('../Dataset/review.json', encoding='utf_8') as review_json_file:
            # loop through all reviews in the existing file and convert to dict
            for review_json in review_json_file:
                review = json.loads(review_json)
                review_txt_file.write(review[u'text'].replace('\n', '\\n') + '\n')
                review_count += 1
    print("""Text from {} restaurant reviews
                written to the new txt file.""".format(review_count))

def unigram():
    # One time use, prepare unigram model
    unigram_sentences_filepath = '../Dataset/unigram_sentences_all.txt'         
    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in lemmatized_sentence_corpus('review_text_all.txt'):
            f.write(sentence + '\n')
    


def bigram():
    # One time use, prepare bigram model
    unigram_sentences = LineSentence('../Dataset/unigram_sentences_all.txt' )
    bigram_model_filepath = '../Models/bigram_model_all'
    bigram_sentences_filepath = '../Dataset/bigram_sentences_all.txt'
    bigram_model = Phrases(unigram_sentences)
    bigram_model.save('../Models/bigram_model_all')
    with codecs.open('../Dataset/bigram_sentences_all.txt', 'w', encoding='utf_8') as f:
        for unigram_sentence in unigram_sentences: 
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])
            f.write(bigram_sentence + '\n')
             
def trigram():
    # One time use, prepare trigram model
    bigram_sentences = LineSentence('../Dataset/bigram_sentences_all.txt')
    trigram_model_filepath = '../Models/trigram_model_all'
    trigram_model = Phrases(bigram_sentences)
    trigram_model.save(trigram_model_filepath)
    trigram_sentences_filepath = '../Dataset/trigram_sentences_all.txt'
    trigram_reviews_filepath = '../Dataset/trigram_transformed_reviews_all.txt'
    with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')
    trigram_sentences = LineSentence(trigram_sentences_filepath)

def insert_trigram_review():
    values0=''
    for i in range(0,len(review)):
        parsed_review = nlp(review['text'][i])
        unigram_review = [token.lemma_.replace('"','') for token in parsed_review
                            if not punct_space(token) and token.lemma_ not in '-PRON-']
        bigram_review = bigram_model[unigram_review]
        trigram_review = trigram_model[bigram_review]
        clean_review = [term for term in trigram_review
                            if term not in spacy.lang.en.stop_words.STOP_WORDS]
        values0 = values0 + '("{}","{}","{}"),'.format(review['review_id'][i],review['business_id'][i],' '.join(clean_review))
        sql = 'insert into parsed_review (review_id, business_id, clean_review) values {};'.format(values0[:-1])
        i += 1
        if i % 100 ==0:
            print(i)
    run_sql(sql)

# create 'trigram_transformed_reviews_all.txt'
def trigram_transform():
    nlp = spacy.load('en_core_web_sm')
    with codecs.open('../Dataset/trigram_transformed_reviews_all.txt', 'w', encoding='utf_8') as f:
        for parsed_review in nlp.pipe(line_review('../Dataset/review_text_all.txt'),
                                        batch_size=10000, n_threads=8):
            # lemmatize the text, removing punctuation and whitespace
            unigram_review = [token.lemma_ for token in parsed_review
                                if not punct_space(token)]
            # apply the first-order and second-order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]
            # remove any remaining stopwords
            trigram_review = [term for term in trigram_review
                                if term not in spacy.lang.en.stop_words.STOP_WORDS]
            # write the transformed review as a line in the new file
            trigram_review = u' '.join(trigram_review)
            f.write(trigram_review + '\n')

# create and save LDA model
def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)

def create_LDA_dict():
    #ONE TIME USE, to create and save LDA model
    trigram_dictionary_filepath = '../Dataset/trigram_dict_all.dict'
    trigram_reviews = LineSentence('../Dataset/trigram_transformed_reviews_all.txt')
    # learn the dictionary by iterating over all of the reviews
    trigram_dictionary = Dictionary(trigram_reviews)
    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
    trigram_dictionary.compactify()
    trigram_dictionary.save(trigram_dictionary_filepath)
    print('LDA dict saved.')
    trigram_bow_filepath = '../Models/trigram_bow_corpus_all.mm'
    MmCorpus.serialize(trigram_bow_filepath, trigram_bow_generator('../Dataset/trigram_transformed_reviews_all.txt'))
    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)
    lda_model_filepath = '../Models/lda_model_all' #lda_model_all_30, lda_model_10topic
    # created LDA model with 10, 30, 50 topics, found 30 has best result
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda = LdaMulticore(trigram_bow_corpus,
                            num_topics=30, #10, 30, 50
                            id2word=trigram_dictionary,
                            workers=8)
    lda.save(lda_model_filepath)
    print('LDA model saved.')

# subtopics
def get_topic_name():
    # return subtopic names
    topic_names_30 = {
        0: u'experience', 1: u'sandwich', 2: u'customer service', 3: u'asian',
        4: u'breakfast', 5: u'discount', 6: u'value', 7: u'pizza', 8: u'burger', 9: u'menu',
        10: u'chinese', 11: u'food quality', 12: u'thai', 13: u'buffet', 14: u'hotel', 15: u'steak',
        16: u'sushi', 17: u'location', 18: u'bar', 19: u'feeling', 20: u'customer service',
        21: u'italian', 22: u'fine dinner', 23: u'dessert', 24: u'wing', 25: u'kid', 26: u'BBQ',
        27: u'mexican', 28: u'price', 29: u'environment'
    }
    return topic_names_30

def lda_show_topic(i = [1]):
    # take list variable, return topic name and sub-topic items
    lda = LdaMulticore.load('../Models/lda_model_all_30')
    name = get_topic_name()
    lst = []
    for x in i:
        print('subtopic = {}'.format(name[x]))
        print(lda.show_topic(x, topn=25))
        lst.append(lda.show_topic(x, topn=25))
    return lst

#load LDA model
lda = LdaMulticore.load('../Models/lda_model_all_30')

def lda_review(txt):
    # given a review, return which sub-topics are included
    trigram_dictionary = Dictionary.load('../Dataset/trigram_dict_all.dict')
    review_bow = trigram_dictionary.doc2bow(txt.split())
    return lda[review_bow]
 
def compare_review(txt1,txt2):
    # compute similarity of two texts
    lda1, lda2 = lda_review(txt1), lda_review(txt2)
    return cosine_similarity(lda1, lda2)

def recom(user_id):
    value = []
    bus_id = business.business_id
    usr_review = get_review_for_person(user_id)
    for i in bus_id:
        get_review_for_buss(i)




# business_id = '8mIrX_LrOnAqWsB5JrOojQ'
def get_review_for_buss(business_id):
    # concat all reviews for given business_id
    sql = """
        select * from parsed_review
        where business_id = '{}'
    """.format(business_id)
    all_review_df = pd.read_sql(sql,conn)
    all_review = all_review_df['clean_review'].str.cat(sep=' ')
    return(all_review)

def get_review_for_person(user_id):
    # concat all reviews for given user_id
    sql = """
        select * from parsed_review
        where user_id = '{}';""".format(user_id)
    all_review_df = pd.read_sql(sql,conn)
    all_review = all_review_df['clean_review'].str.cat(sep=' ')
    return(all_review)

def get_buss_attr(business_id):
    # compute subtopic scores given all reviews from specific business_id
    review_buss = get_review_for_buss(business_id)
    lda_rst = lda_review(review_buss)
    sql = "select stars from business where business_id = '{}'".format(business_id)
    avg_star = pd.read_sql(sql,conn)['stars'].values[0]
    buss_attr = [business_id,avg_star]
    lst = [0]*30
    for i in lda_rst:
        lst[i[0]] = i[1]
    buss_attr.extend(lst)
    return buss_attr

def get_user_attr(user_id):
    # compute subtopic scores given all reviews from specific user_id
    review_user = get_review_for_person(user_id)
    lda_rst = lda_review(review_user)
    sql = "select average_stars from user where user_id = '{}'".format(user_id)
    avg_star = pd.read_sql(sql,conn)['average_stars'].values[0]
    user_attr = [user_id,avg_star]
    lst = [0]*30
    for i in lda_rst:
        lst[i[0]] = i[1]
    user_attr.extend(lst)
    return user_attr


sql = "select count(*) from business where city like '%Vegas%' and categories like '%estaurant%'"

sql = "select count(*) as restaurants, count(distinct(user_id)) from review where business_id in (select business_id from business where city like '%Vegas%' and categories like '%estaurant%')"
a = pd.read_sql(sql,conn)
