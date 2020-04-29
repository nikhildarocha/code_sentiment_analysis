#!/usr/local/bin/python3
import os
import json
import tweepy
import csv

consumer_key = os.getenv('CONSUMER_KEY_TWITTER')
consumer_secret = os.getenv('CONSUMER_SECRET_TWITTER')
access_key = os.getenv('ACCESS_KEY_TWITTER')
access_secret = os.getenv('ACCESS_SECRET_TWITTER')

def get_all_tweets(screen_name):
  auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_key, access_secret)
  api = tweepy.API(auth)
  alltweets = []	
  new_tweets = api.user_timeline(screen_name = screen_name,count=200)
  alltweets.extend(new_tweets)
  oldest = alltweets[-1].id - 1
  while len(new_tweets) > 0:
    print("getting tweets before %s" % (oldest))
    new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    print("...%s tweets downloaded so far" % (len(alltweets)))
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
    with open('%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)
        pass
if __name__ == '__main__':
    get_all_tweets("@Barclaycard")