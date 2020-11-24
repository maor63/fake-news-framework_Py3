# Created by aviade      
# Time: 15/05/2016 08:48

import logging
from logging import config

from DB.schema_definition import DB
from Twitter_API.twitter_api_requester import TwitterApiRequester
from configuration.config_class import getConfig
from commons.commons import *
from datetime import datetime, timedelta
from email.utils import parsedate_tz
import datetime
from functools import reduce


def save_recent_tweets(tweets, keyword, db):
    for tweet in tweets:
        db.insert_json_recent_post(tweet, keyword)
    db.session.commit()


def save_popular_tweets(popular_tweets, keyword, db):
    '''
    popular_tweet = popular_tweets[0]
    popular_tweet_id = popular_tweet.id
    recent_post = db.get_recent_post_by_id(popular_tweet_id)
    #if recent_post is not None:
    db.insert_json_popular_post(popular_tweet)
    db.session.commit()
    '''

    for popular_tweet in popular_tweets:
        popular_tweet_id = popular_tweet.id
        recent_post = db.get_recent_post_by_id(popular_tweet_id)
        if recent_post is not None:
            db.insert_json_popular_post(popular_tweet, keyword)
    db.session.commit()


def are_recent_posts_exist_by_term(term):
    recent_posts_by_term = db.get_json_recent_posts_by_term(term)
    return len(recent_posts_by_term) > 0

def convert_diff_to_date(diff, now_date):
    original_date = now_date - diff

    original_date_created_at = date_to_str(original_date)
    return original_date_created_at


def get_post_by_date(date, popular_posts):
    for popular_post in popular_posts:
        created_at = popular_post.created_at
        if created_at == date:
            return popular_post
    return None

def calculateAverageCreationTime2(popular_posts, now_date):
    print(("Number of popular posts ais: " + str(len(popular_posts))))
    dates_diff = []
    for popular_post in popular_posts:
        created_at = popular_post.created_at
        id = popular_post.id
        print(("Popular post id = " + str(id) + " created at " + created_at))

        date = to_datetime(created_at)
        diff = now_date - date
        dates_diff.append(diff)

    min_diff = min(dates_diff)
    min_date = convert_diff_to_date(min_diff, now_date)
    print(("The minimal date for a post was created at " + str(min_date)))
    print("-------------")


    max_diff = max(dates_diff)
    max_date = convert_diff_to_date(max_diff, now_date)
    print(("The maximal date for a post was created at " + str(max_date)))
    print("-------------")

    average_diff = reduce(lambda x, y: x + y, dates_diff) / len(dates_diff)
    average_date = convert_diff_to_date(average_diff, now_date)
    print(("The average post was created at " + str(average_date)))
    print("-------------")


def calculateAverageCreationTime(popular_posts, now_date):
    print(("Number of popular posts is: " + str(len(popular_posts))))
    dates_diff = []
    for popular_post in popular_posts:
        created_at = popular_post.created_at
        id = popular_post.id
        print(("Popular post id = " + str(id) + " created at " + created_at))

        date = to_datetime(created_at)
        diff = now_date - date
        dates_diff.append(diff)

    min_diff = min(dates_diff)
    min_date = convert_diff_to_date(min_diff, now_date)
    print(("The minimal date for a post was created at " + str(min_date)))
    print("-------------")


    max_diff = max(dates_diff)
    max_date = convert_diff_to_date(max_diff, now_date)
    print(("The maximal date for a post was created at " + str(max_date)))
    print("-------------")

    average_diff = reduce(lambda x, y: x + y, dates_diff) / len(dates_diff)
    average_date = convert_diff_to_date(average_diff, now_date)
    print(("The average post was created at " + str(average_date)))
    print("-------------")



def to_datetime(str_date):
    time_tuple = parsedate_tz(str_date.strip())
    dt = datetime.datetime(*time_tuple[:6])
    return dt - timedelta(seconds=time_tuple[-1])


def find_post_by_date(given_date, created_at_date_dictionry, created_at_post_dictionry):
    created_at = list(created_at_date_dictionry.keys())[list(created_at_date_dictionry.values()).index(given_date)]
    if created_at is not None:
        post = created_at_post_dictionry[created_at]
        return post

    '''
    found_created_at = None
    for created_at, created_at_date in created_at_date_dictionry:
        if created_at_date == given_date:
            found_created_at = created_at
    if found_created_at is not None:
        post = created_at_post_dictionry[found_created_at]
        return post
    '''

def analyze_posts(popular_posts):
    now_date = datetime.datetime.now()
    print(("Number of popular posts ais: " + str(len(popular_posts))))
    dates_diff = []
    created_at_post_dictionry = {}
    created_at_date_dictionry = {}
    for popular_post in popular_posts:
        created_at = popular_post.created_at
        id = popular_post.id
        print(("Popular post id = " + str(id) + " created at " + created_at))

        created_at_post_dictionry[created_at] = popular_post
        date = to_datetime(created_at)
        created_at_date_dictionry[created_at] = date

        diff = now_date - date
        dates_diff.append(diff)

    creation_dates = list(created_at_date_dictionry.values())

    min_creation_date = min(creation_dates)
    print(("The minimal date for a post was created at " + str(min_creation_date)))

    min_post = find_post_by_date(min_creation_date, created_at_date_dictionry, created_at_post_dictionry)
    print(("The minimal post favorite_count is: " + str(min_post.favorite_count)))
    print(("The minimal post retweet_count is: " + str(min_post.retweet_count)))
    print("-------------")

    max_creation_date = max(creation_dates)
    print(("The maximal date for a post was created at " + str(max_creation_date)))

    max_post = find_post_by_date(max_creation_date, created_at_date_dictionry, created_at_post_dictionry)
    print(("The minimal post favorite_count is: " + str(max_post.favorite_count)))
    print(("The minimal post retweet_count is: " + str(max_post.retweet_count)))
    print("-------------")

    average_diff = reduce(lambda x, y: x + y, dates_diff) / len(dates_diff)
    average_date = convert_diff_to_date(average_diff, now_date)
    print(("The average post was created at " + str(average_date)))
    print("-------------")





    '''
    max_diff = max(dates_diff)
    max_date = convert_diff_to_date(max_diff, now_date)
    print("The maximal date for a post was created at " + str(max_date))
    print("-------------")

    average_diff = reduce(lambda x, y: x + y, dates_diff) / len(dates_diff)
    average_date = convert_diff_to_date(average_diff, now_date)
    print("The average post was created at " + str(average_date))
    print("-------------")
    '''

if __name__ == '__main__':
    config_parser = getConfig()
    logging.config.fileConfig(config_parser.get("Logger", "logger_conf_file"))
    logging.info("Start program...")
    print("Start program...")

    db = DB()
    db.setUp()

    logging.info("Creating TwitterApiRequester")
    print("Creating TwitterApiRequester")
    twitter_api_requester = TwitterApiRequester()

    keywords_line = config_parser.get("PostDetector", "keywords")
    keywords = keywords_line.split(",")

    for keyword in keywords:
        keyword = keyword.translate(None, '"\"').strip()
        are_recent_posts = are_recent_posts_exist_by_term(keyword)
        if are_recent_posts is False:
            recent_tweets = twitter_api_requester.get_tweets_by_term(keyword, "recent")
            save_recent_tweets(recent_tweets, keyword, db)

        popular_tweets = twitter_api_requester.get_tweets_by_term(keyword, "popular")

        now = datetime.datetime.now()
        print(("Now = " + str(now)))
        print("-------------------")
        analyze_posts(popular_tweets)

        #calculateAverageCreationTime(popular_tweets, now)

        save_popular_tweets(popular_tweets, keyword, db)

    #term = "'Online TV'"
    #term = "Online TV"
    #term = "'Internet TV'"
    #term = "Internet TV"
    #term = "'Smart TV'"
    #term = "Smart TV"



