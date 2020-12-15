#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#########################################################################
# File Name: connect_mongo.py
# Author: NLP_zhaozhenyu
# Mail: zhaozhenyu_tx@126.com
# Created Time: 10:42:41 2020-04-15
#########################################################################
import sys
import pymongo
import json
import urllib.request
import urllib.parse
import read_from_txt
from pymongo import MongoClient
import datetime



def get_highcheck():
    host = '172.31.30.178'#高展审核
    db_location = 'content' #高展审核
    collection_location = 'top_checked' #高展审核
    d = datetime.datetime(2020,4,7)

    client = MongoClient(host,27017)
    db = client[db_location]
    collection = db[collection_location]
    document = collection.find({'$or':[{"type":"adult", "review":{"$exists":"true"}}, \
    {"review.reason":"nudity and sexual content"}]})#,"review.dt":{"$gt":"2020-04-26"}}) #高展审核
    docids = set()
    docids_dict = {}
    
    for k in document:
        info = k.get('review')
        print(str(info)+'\t'+json.dumps(k['doc']))

def get_offline_from_url(url,key=['_id','url','stitle','seg_content']):
    
    '''
    client = MongoClient('rs-offline.mongo.nb.com:27017', replicaset='rs-offline', \
        readPreference='secondaryPreferred',unicode_decode_error_handler='ignore')['news']['data']
    '''
    client = MongoClient('172.31.27.159',27017,unicode_decode_error_handler='ignore')['news']['data']
    jd = client.find_one({'url':url})
    if jd!=None:
        cur = {}
        for k in key:
            cur[k] = jd.get(k)
        return cur
    else:
        return None

def get_offline_from_docid(_id,key=['_id','url','stitle','seg_content']):
    
    '''
    client = MongoClient('rs-offline.mongo.nb.com:27017', replicaset='rs-offline', \
        readPreference='secondaryPreferred',unicode_decode_error_handler='ignore')['news']['data']
    '''
    client = MongoClient('172.31.27.159',27017,unicode_decode_error_handler='ignore')['news']['data']
    jd = client.find_one({'_id':_id})
    if jd!=None:
        cur = {}
        for k in key:
            cur[k] = jd.get(k)
        return cur
    else:
        return None           

def get_staticfeature_from_docid(_id,key=['_id','url','stitle','seg_content']):
    client = MongoClient('172.31.29.170',27017,unicode_decode_error_handler='ignore')['staticFeature']['document']
    jd = client.find_one({'_id':_id})
    if jd!=None:
        cur = {}
        for k in key:
            cur[k] = jd.get(k)
        return cur
    else:
        return None    

def sample_highquality_from_offline(start_time, num, quality=[4,5],key=['_id','url','seg_title','seg_content', 'domain']):
    #start_time='2020-06-01'
    #staticfeature = MongoClient('172.31.29.170',27017,unicode_decode_error_handler='ignore')['staticFeature']['document']
    offline = MongoClient('172.31.27.159',27017,unicode_decode_error_handler='ignore')['news']['data']
    quality_mongo = MongoClient('172.24.22.248',27017,unicode_decode_error_handler='ignore')['documentLabels']['sourceInfo']
    documents = offline.aggregate([{'$match':{'insert_time':{'$gte':start_time}}},\
        {'$sample':{'size':num}}],allowDiskUse=True)
    res = []
    for doc in documents:
        cur = {}
        for k in key:
            cur[k] = doc.get(k)
        domain = doc.get('domain','')
        dom = quality_mongo.find_one({'domain':domain})
        if dom!=None and 'quality' in dom:
            if dom['quality'] in quality:
                cur['quality'] =  dom['quality']
                res.append(cur)
    return res      
        
def get_staticfeature_from_url(url,key= ['_id','url','stitle','seg_content']):
    offline_res  = get_offline_from_url(url,key = ['_id'])
    res = {}
    if offline_res !=None:
        online_res = get_staticfeature_from_docid(offline_res['_id'],key = key)
        if online_res !=None:
            return online_res
    return None



if __name__ == '__main__':
    print(get_staticfeature_from_url('https://www.washingtonpost.com/podcasts/post-reports/the-attorney-generals-defense/'))#sample_highquality_from_offline()
    #get_staticfeature_from_docid(sys.argv[1])
    #get_offline_from_url(sys.argv[1])
    #get_highcheck()
