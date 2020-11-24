# encoding: utf-8
# need to be added to the system



import logging

from DB.schema_definition import Post, Author, Politifact_Liar_Dataset, date
from commons.commons import compute_post_guid, compute_author_guid_by_author_name, extract_tweet_publiction_date, \
    str_to_date
from commons.method_executor import Method_Executor
from preprocessing_tools.post_importer import PostImporter
import csv
import os
import pandas as pd

from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api


class US_2016_Presidential_Election_Importer(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._data_folder = self._config_parser.eval(self.__class__.__name__, "data_folder")

        self._social_network_crawler = Twitter_Rest_Api(db)

    def retrieve_tweets_from_scratch(self):
        file_names = os.listdir(self._data_folder)

        for file_name in file_names:
            lines = [line.rstrip('\n') for line in open(self._data_folder + file_name)]

            num_of_tweet_ids = len(lines)
            msg = "\r Number of tweets ids left to retrieve is: {0}".format(num_of_tweet_ids)
            print(msg, end="")

            self._social_network_crawler.get_tweets_by_tweet_ids_and_add_to_db(lines)


    def continue_retrieving_tweets_in_case_of_crush(self):
        file_names = os.listdir(self._data_folder)
        for file_name in file_names:
            total_tweet_ids = [line.rstrip('\n') for line in open(self._data_folder + file_name)]
            total_tweet_ids = set(total_tweet_ids)

            already_found_tweets_ids_tuples = self._db.get_post_osn_ids()
            already_found_tweets_ids = [tweets_ids_tuple[0] for tweets_ids_tuple in already_found_tweets_ids_tuples]
            already_found_tweets_ids = set(already_found_tweets_ids)

            left_to_retrieve_tweet_ids = total_tweet_ids - already_found_tweets_ids
            left_to_retrieve_tweet_ids = list(left_to_retrieve_tweet_ids)
            num_of_tweet_ids_to_retrieve = len(left_to_retrieve_tweet_ids)
            msg = "\r Number of tweets ids left to retrieve is: {0}".format(num_of_tweet_ids_to_retrieve)
            print(msg, end="")

            self._social_network_crawler.get_tweets_by_tweet_ids_and_add_to_db(left_to_retrieve_tweet_ids)
