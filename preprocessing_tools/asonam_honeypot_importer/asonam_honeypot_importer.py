import csv
from collections import defaultdict
from os import listdir
from preprocessing_tools.abstract_controller import AbstractController
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api


class AsonamHoneypotImporter(AbstractController):
    def __init__(self, db):
        super(AsonamHoneypotImporter, self).__init__(db)
        self.twitter_rest_api = Twitter_Rest_Api(self._db)
        self._data_path = self._config_parser.eval(self.__class__.__name__, "data_path")

    def execute(self, window_start=None):
        all_sub_files = listdir(self._data_path)
        for data_file in all_sub_files:
            self.add_tweet_data_to_db(data_file)

    def add_tweet_data_to_db(self, data_file):
        type_to_users_tweets_dict = self.load_file(self._data_path + data_file)
        for author_type in type_to_users_tweets_dict:
            tweets_ids = []
            for author in type_to_users_tweets_dict[author_type]:
                tweets_ids.extend(type_to_users_tweets_dict[author_type][author])
            tweets = self.get_tweets(tweets_ids, str(author_type))
            self.twitter_rest_api._save_posts_and_authors(tweets, str(author_type))

    def load_file(self, file_path):
        file = open(file_path, 'r', newline='')
        csv_file = csv.reader(file, delimiter=' ')
        type_to_users_tweets = defaultdict(dict)
        for row in csv_file:
            author_type = row[0]
            author_id = row[1]
            tweet_ids = row[2:]
            type_to_users_tweets[author_type][author_id] = tweet_ids
        file.close()
        return type_to_users_tweets

    def get_tweets(self, tweets_ids, author_type=""):
        return self.twitter_rest_api.get_tweets_by_ids(tweets_ids, author_type)
