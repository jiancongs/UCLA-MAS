import os
import codecs
import json
import spacy
import pandas as pd
import itertools as it
import sqlite3

data_directory = os.path.join('/Users/erics/Documents/Documents/UCLA/Thesis/Dataset')
businesses_filepath = os.path.join(data_directory,'business.json')
review_filepath = os.path.join(data_directory,'review.json')
conn = sqlite3.connect('../Dataset/yelp.db')

# with codecs.open(businesses_filepath, encoding='utf_8') as f:
#     first_review_record = f.readline() 
#     print(first_review_record)

#data overview
with codecs.open('../Dataset/business.json', encoding='utf_8') as f:
    first_business_record = f.readline() 
    print(first_business_record)
"""
business:
business_id, name, address, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes[],categories,hours
"""

with codecs.open('../Dataset/review.json', encoding='utf_8') as f:
    first_review_record = f.readline()
    print(first_review_record)
"""
review:
review_id, user_id, business_id, stars. usful, funny, cool, text, date
"""

with codecs.open('../Dataset/user.json', encoding='utf_8') as f:
    first_user_record = f.readline()
    print(first_user_record)
"""
user:
user_id, name, review_count, yelping_since, useful, funny, cool, elite, friends(list of user_id), fans, average_stars, compliment_hot, compliment_more,
compliment_profile, compliment_cute, compliment_list, compliment_note, compliment_plain, compliment_cool, compliment_funny, compliment_writer, compliment_photos
"""

with codecs.open('../Dataset/tip.json', encoding='utf_8') as f:
    first_tip_record = f.readline()
    print(first_tip_record)
"""
tip:
user_id, business_id, text, compliment_count
"""

# Design and create database
"""
refer to create_table.sql in Dateset folder
"""


def run_sql(sql):
    with sqlite3.connect('../Dataset/yelp.db') as con:
        c = con.cursor()
        c.execute(sql)
        con.commit()

# read, reformat and insert data into db
def insert_business():
    print("inserting business data into database")
    count = 0
    with codecs.open('../Dataset/business.json', encoding='utf_8') as f:
        for business in f:
            t = json.loads(business)
            # if u'Restaurants' not in business[u'categories']:
            lst = t['business_id'], t['name'], t['address'],t['city'], t['state'], t['postal_code'], t['latitude'], t['longitude'], t['stars'], t['review_count'], t['is_open'], str(t['attributes']), str(t['categories']), str(t['hours'])
            lst = [str.replace(x, "\'", "") if type(x) == str else x for x in lst]
            sql = "INSERT INTO business (business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours) values ({});".format(str(lst).strip('[]'))
            # print(sql)
            try:
                run_sql(sql)
            except:
                print(sql)
                break
            count += 1
    print('inserted {} recoreds into business table.'.format(count))

def insert_review():
    print("inserting review data into database")
    count = 0
    with codecs.open('../Dataset/review.json', encoding='utf_8') as f:
        for review in f:
            t = json.loads(review)
            text = " ".join(t['text'].split("\n"))
            lst = (t['review_id'], t['user_id'], t['business_id'], t['stars'], t['useful'], text, t['date'])
            lst = [str.replace(x, "\'", "") if type(x) == str else x for x in lst]
            sql = "INSERT INTO review (review_id, user_id, business_id, stars, useful, text, date) values ({});".format(str(lst).strip('[]'))
            # print(sql)
            try:
                run_sql(sql)
            except:
                print(sql)
                break
            count += 1
    print('inserted {} recoreds into review table.'.format(count))

def insert_user():
    print("inserting user data into database")
    count = 0
    with codecs.open('../Dataset/user.json', encoding='utf_8') as f:
        for user in f:
            t = json.loads(user)
            lst = (t['user_id'], t['name'], t['review_count'], t['yelping_since'], t['useful'], t['funny'], t['cool'], t['elite'],t['average_stars'])
            lst = [str.replace(x, "\'", "") if type(x) == str else x for x in lst]
            sql = "INSERT INTO user (user_id, name, review_count, yelping_since, useful, funny, cool, elite, average_stars) values ({});".format(str(lst).strip('[]'))
            # print(sql)
            try:
                run_sql(sql)
            except:
                print(sql)
                break
            count += 1
    print('inserted {} recoreds into insert_user table.'.format(count))

def insert_tip():
    print("inserting tip data into database")
    count = 0
    with codecs.open('../Dataset/tip.json', encoding='utf_8') as f:
        for tip in f:
            t = json.loads(tip)
            lst = (t['user_id'], t['business_id'], t['text'])
            lst = [str.replace(x, "\'", "") if type(x) == str else x for x in lst]
            sql = "INSERT INTO tip (user_id, business_id, text) values ({});".format(str(lst).strip('[]'))
            # print(sql)
            try:
                run_sql(sql)
            except:
                print(sql)
                break
            count += 1
    print('inserted {} recoreds into tip table.'.format(count))


def insert_all():
    print("inserting Yelp data into database, this will take a long time since the data is huge...")
    insert_business()
    insert_user()
    insert_review()
    insert_tip()
