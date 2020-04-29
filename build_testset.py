#!/usr/local/bin/python3
import os
import json
import tweepy
import csv


consumer_key = os.getenv('CONSUMER_KEY_TWITTER')
consumer_secret = os.getenv('CONSUMER_SECRET_TWITTER')
access_key = os.getenv('ACCESS_KEY_TWITTER')
access_secret = os.getenv('ACCESS_SECRET_TWITTER')

# initialize api instance
auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
twitter_api = tweepy.API(auth)

# test authentication
print(twitter_api.verify_credentials())


def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.search(search_keyword, count = 100)
        
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None
    

search_term = input("Enter a search keyword:")
testDataSet = buildTestSet(search_term)

print(testDataSet[0:4])