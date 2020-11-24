
import datetime
from collections import defaultdict
from DB.schema_definition import Claim, Post, Claim_Tweet_Connection
from commons.commons import compute_post_guid, date_to_str, compute_author_guid_by_author_name
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd
import os
import numpy as np

from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api


class ComLexClaimTweetImporter(AbstractController):

    def __init__(self, db):
        super(ComLexClaimTweetImporter, self).__init__(db)
        self._claims_csv_path = self._config_parser.eval(self.__class__.__name__, "claims_csv_path")
        self._tweets_csv_path = self._config_parser.eval(self.__class__.__name__, "tweets_csv_path")
        self._max_posts_without_save = self._config_parser.eval(self.__class__.__name__, "max_posts_without_save")
        self._claims = []
        self._posts = []
        self._claim_tweet_connections = []
        self._author_features = []
        self.posts_csv_df = pd.DataFrame()

    def execute(self, window_start):
        self._import_claims()
        self._import_tweets()

    def _import_claims(self):
        claim_csv_df = pd.DataFrame.from_csv(self._claims_csv_path)
        # twitter_claims_df = claim_csv_df[claim_csv_df['site'] == u'twitter']
        claim_csv_df.reset_index(inplace=True)
        claim_csv_df.apply(self._convert_row_to_claim, axis=1)
        self._db.addPosts(self._claims)
        self._db.add_author_features(self._author_features)
        self._claims = []
        self._author_features = []
        pass

    def _convert_row_to_claim(self, claim_row):
        claim = Claim()
        claim.claim_id = compute_author_guid_by_author_name(str(claim_row['social_id']))
        claim.domain = str(claim_row['site'])
        claim.verdict = str(claim_row['ruling_val'])
        claim.verdict_date = datetime.datetime.fromtimestamp(claim_row['ruling_time'])
        self._claims.append(claim)

        attribute_name = self.__class__.__name__ + "_claim_type"
        author_feature = BaseFeatureGenerator.create_author_feature(attribute_name, claim.claim_id, claim.verdict,
                                                                    self._window_start, self._window_end)
        self._author_features.append(author_feature)
        attribute_name = self.__class__.__name__ + "_claim_id"
        author_feature = BaseFeatureGenerator.create_author_feature(attribute_name, claim.claim_id, claim.claim_id,
                                                                    self._window_start, self._window_end)
        self._author_features.append(author_feature)

    def _import_tweets(self):
        for posts_file in os.listdir(self._tweets_csv_path):
            print("import {0} file".format(posts_file))
            self._current_row = 1
            self.posts_csv_df = pd.DataFrame.from_csv(self._tweets_csv_path + "/{0}".format(posts_file))
            self.posts_csv_df.reset_index(inplace=True)
            self.posts_csv_df.apply(self._convert_row_to_post, axis=1)
            self._save_posts_and_connections()
            print()

    def _save_posts_and_connections(self):
        self._db.addPosts(self._posts)
        self._db.addPosts(self._claim_tweet_connections)
        self._posts = []
        self._claim_tweet_connections = []

    def _convert_row_to_post(self, row):
        # [site, social_id, username_hash, comment_time, comment_tokens]
        print("\rInsert post to DataFrame {0}/{1}".format(self._current_row, len(self.posts_csv_df)), end="")
        self._current_row += 1
        date = datetime.datetime.fromtimestamp(row['comment_time'])
        post = Post()
        claim_id = compute_author_guid_by_author_name(str(row['social_id']))
        post.post_id = str(
            compute_post_guid(row['site'] + str(claim_id), row['username_hash'], date_to_str(date)))
        post.content = str(row['comment_tokens'])
        post.author = str(row['username_hash'])
        post.author_guid = str(row['username_hash'])
        post.domain = str(row['site'])
        post.date = date
        self._posts.append(post)

        claim_tweet_connection = Claim_Tweet_Connection()
        claim_tweet_connection.claim_id = str(claim_id)
        claim_tweet_connection.post_id = str(post.post_id)
        self._claim_tweet_connections.append(claim_tweet_connection)

        if self._current_row % self._max_posts_without_save == 0:
            self._save_posts_and_connections()
