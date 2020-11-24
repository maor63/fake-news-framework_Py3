

from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.abstract_controller import AbstractController
import json
import pandas as pd
import os

__author__ = "Aviad Elyashar"

############################################################################################################
# This module is responsible to generate features for topics from TopicDistrobutionVisualizationGenerator
# according to the predictions of the given classifier.
#
# Instructions:
# 1. Run PreprocessVisualizationData - convert the claims in the posts table to topics.
# 2. Train a classifier seperately on different dataset. For example, Train on Virtual TV Manually Labeled
#    Account dataset a RandomForest classifier based on 10 features.
#    Output: pkl file of the trained classifier and pkl of the 10 feature names.
# 3. Use the ExperimentalEnvironment on the targeted dataset and predict with the trained classifier on the
#    unlabeled authors in the targeted dataset.
#    Input: pkl of the trained classifer.
#    Change in the config.in under the module titled ExperimentalEnvironment put the selected features
#    Output: Predictions of the unlabeled authors in table in the DB and CSV file.
# 4. Use the TopicDistrobutionVisualizationGenerator in order to create visualization.
#    Output: stasitics file placed in the output directory under the folder of the current module.
# 5. Use this module. It requires the statistics file of the predictions
##############################################################################################################
class ClaimTopicFeatureGenerator(AbstractController):
    def __init__(self, db, **kwargs):
        AbstractController.__init__(self, db)
        self._input_path = self._config_parser.eval(self.__class__.__name__, "input_path")
        self._topic_statistics_file = self._config_parser.eval(self.__class__.__name__, "topic_statistics_file")
        self._post_id_topic_id_dict_file = self._config_parser.eval(self.__class__.__name__, "post_id_topic_id_dict_file")
        self._classifier_name = self._config_parser.eval(self.__class__.__name__, "classifier_name")
        #self._dataset_name = self._config_parser.eval(self.__class__.__name__, "dataset_name")
        self._selected_features_file = self._config_parser.eval(self.__class__.__name__, "selected_features_file")
        self._features_extracted_by_tuple = self._config_parser.eval(self.__class__.__name__, "features_extracted_by_tuple")
        self._aggregated_features = self._config_parser.eval(self.__class__.__name__, "aggregated_features")

        self._prefix = self.__class__.__name__

    def execute(self, window_start=None):

        self._claim_id_claim_type_dict = self._create_claim_id_claim_type_dictionary()

        directory_names = os.listdir(self._input_path)
        for dataset_name in directory_names:
            self._dataset_name = dataset_name
            #with or without retweets
            aggregation_retweets_type_directories = os.listdir(self._input_path + self._dataset_name)
            for aggregation_retweets_type in aggregation_retweets_type_directories:
                self._aggregation_retweets_type = aggregation_retweets_type
                target_path = "{0}/{1}/{2}/".format(self._input_path, self._dataset_name, aggregation_retweets_type)

                # read post_id_topic_id_dict_file
                with open(target_path + self._post_id_topic_id_dict_file) as file:
                    self._post_id_topic_id_dict = json.load(file)


                self._topic_id_post_id_dict = {value: key for key, value in self._post_id_topic_id_dict.items()}

                self._topic_statistics_df = pd.read_csv(target_path + self._topic_statistics_file)

                author_features = []
                for index, row in self._topic_statistics_df.iterrows():
                    df_tuple = tuple(row)
                    for i, feature_name in enumerate(self._features_extracted_by_tuple):
                        msg = "\rCalculating features: [{0}/{1}: {2} {3} {4}]]".format(i, len(self._features_extracted_by_tuple),
                                                                                       feature_name, self._dataset_name,
                                                                                       self._aggregation_retweets_type)
                        print(msg, end="")

                        post_id, attribute_name, attribute_value = getattr(self, feature_name)(df_tuple)
                        author_feature = BaseFeatureGenerator.create_author_feature(attribute_name, post_id, attribute_value,
                                                                                    self._window_start, self._window_end)
                        author_features.append(author_feature)

                for j, aggregated_feature in enumerate(self._aggregated_features):
                    msg = "\rCalculating aggregated features: [{0}/{1} {2} {3} {4}]]".format(j, len(self._aggregated_features),
                                                                                             aggregated_feature, self._dataset_name,
                                                                                       self._aggregation_retweets_type)
                    print(msg, end="")

                    for topic_id, post_id in list(self._topic_id_post_id_dict.items()):
                        attribute_value = getattr(self, aggregated_feature)(topic_id)
                        attribute_name = "{0}_{1}_{2}_{3}_{4}".format(self._prefix, aggregated_feature, self._dataset_name,
                                                                                        self._aggregation_retweets_type,
                                                                                        self._classifier_name)
                        author_feature = BaseFeatureGenerator.create_author_feature(attribute_name, post_id, attribute_value,
                                                                                    self._window_start, self._window_end)
                        author_features.append(author_feature)

                self._db.add_author_features(author_features)



    def authors_in_bucket(self, topic_tuple):
        topic_number = int(topic_tuple[0])
        post_id = self._topic_id_post_id_dict[topic_number]

        bucket = topic_tuple[1]
        num_of_authors = int(topic_tuple[2])
        attribute_name = "{0}_authors_in_bucket_{1}_dataset_{2}_{3}_classifier_{4}".format(self._prefix, bucket,
                                                                                            self._dataset_name,
                                                                                        self._aggregation_retweets_type,
                                                                                        self._classifier_name)

        return post_id, attribute_name, num_of_authors

    def posts_in_bucket(self, topic_tuple):
        topic_number = int(topic_tuple[0])
        post_id = self._topic_id_post_id_dict[topic_number]

        bucket = topic_tuple[1]

        num_of_posts = int(topic_tuple[3])
        attribute_name = "{0}_posts_in_bucket_{1}_dataset_{2}_{3}_classifier_{4}".format(self._prefix, bucket,
                                                                                          self._dataset_name,
                                                                                          self._aggregation_retweets_type,
                                                                                          self._classifier_name)

        return post_id, attribute_name, num_of_posts


    def post_author_ratio(self, topic_tuple):
        topic_number = int(topic_tuple[0])
        post_id = self._topic_id_post_id_dict[topic_number]

        bucket = topic_tuple[1]

        post_author_ratio = topic_tuple[4]
        attribute_name = "{0}_post_author_ratio_in_bucket_{1}_dataset_{2}_{3}_classifier_{4}".format(self._prefix,
                                                                                                      bucket,
                                                                                                      self._dataset_name,
                                                                                                      self._aggregation_retweets_type,
                                                                                                      self._classifier_name)

        return post_id, attribute_name, post_author_ratio

    def topic_type(self, topic_tuple):
        topic_number = int(topic_tuple[0])
        post_id = self._topic_id_post_id_dict[topic_number]
        claim_type = self._claim_id_claim_type_dict[post_id]
        attribute_name = self._prefix + "_topic_type"
        return post_id, attribute_name, claim_type

    def _create_claim_id_claim_type_dictionary(self):
        claim_id_claim_type_tuples = self._db.get_posts_by_selected_domain(self._domain)
        claim_id_claim_type_dict = {}
        for claim_id_claim_type_tuple in claim_id_claim_type_tuples:
            claim_id = claim_id_claim_type_tuple[0]
            claim_type = claim_id_claim_type_tuple[1]
            if claim_type == "mostly-false" or claim_type == "barely-true" or claim_type == "pants-fire" or \
                            claim_type == 'false':
                claim_type = "FALSE"
            elif claim_type == "mostly-true" or claim_type == "true":
                claim_type = "TRUE"
            claim_id_claim_type_dict[claim_id] = claim_type
        return claim_id_claim_type_dict

    def _get_max_authors_row(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        df_group_by_topic = df_group_by_topic.reset_index()
        max_authors_row = df_group_by_topic.iloc[df_group_by_topic.authors.argmax()].tolist()
        return max_authors_row

    def _get_min_authors_row(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        df_group_by_topic = df_group_by_topic.reset_index()
        min_authors_row = df_group_by_topic.iloc[df_group_by_topic.authors.argmin()].tolist()
        return min_authors_row

    def _get_max_posts_row(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        df_group_by_topic = df_group_by_topic.reset_index()
        max_posts_row = df_group_by_topic.iloc[df_group_by_topic.posts.argmax()].tolist()
        return max_posts_row

    def _get_min_posts_row(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        df_group_by_topic = df_group_by_topic.reset_index()
        min_posts_row = df_group_by_topic.iloc[df_group_by_topic.posts.argmin()].tolist()
        return min_posts_row

    #################################
    # Max authors
    ##################################
    def max_authors_bucket(self, topic_id):
        max_authors_row = self._get_max_authors_row(topic_id)
        bucket = max_authors_row[2]

        return bucket

    def max_num_of_authors(self, topic_id):
        max_authors_row = self._get_max_authors_row(topic_id)
        max_num_of_authors = int(max_authors_row[3])

        return max_num_of_authors

    def max_authors_num_of_posts(self, topic_id):
        max_authors_row = self._get_max_authors_row(topic_id)
        num_of_posts_when_max_authors = int(max_authors_row[4])

        return num_of_posts_when_max_authors

    def max_authors_post_author_ratio(self, topic_id):
        max_authors_row = self._get_max_authors_row(topic_id)
        post_author_ratio_when_max_authors = max_authors_row[5]

        return post_author_ratio_when_max_authors

    #################################
    # Min authors
    ##################################


    def min_authors_bucket(self, topic_id):
        min_authors_row = self._get_min_authors_row(topic_id)
        bucket = min_authors_row[2]

        return bucket

    def min_num_of_authors(self, topic_id):
        min_authors_row = self._get_min_authors_row(topic_id)
        min_num_of_authors = int(min_authors_row[3])

        return min_num_of_authors

    def min_authors_num_of_posts(self, topic_id):
        min_authors_row = self._get_min_authors_row(topic_id)
        num_of_posts_when_min_authors = int(min_authors_row[4])

        return num_of_posts_when_min_authors

    def min_authors_post_author_ratio(self, topic_id):
        min_authors_row = self._get_min_authors_row(topic_id)
        post_author_ratio_when_min_authors = min_authors_row[5]

        return post_author_ratio_when_min_authors

    #################################
    # Max posts
    ##################################

    def max_posts_bucket(self, topic_id):
        max_posts_row = self._get_max_posts_row(topic_id)
        bucket = max_posts_row[2]

        return bucket

    def max_num_of_posts(self, topic_id):
        max_posts_row = self._get_max_posts_row(topic_id)
        max_num_of_posts = int(max_posts_row[4])

        return max_num_of_posts

    def max_posts_num_of_authors(self, topic_id):
        max_posts_row = self._get_max_posts_row(topic_id)
        num_of_authors_when_max_posts = int(max_posts_row[3])

        return num_of_authors_when_max_posts

    def max_posts_post_author_ratio(self, topic_id):
        max_posts_row = self._get_max_posts_row(topic_id)
        post_author_ratio_when_max_posts = max_posts_row[5]

        return post_author_ratio_when_max_posts

    #################################
    # Min posts
    ##################################

    def min_posts_bucket(self, topic_id):
        min_posts_row = self._get_min_posts_row(topic_id)
        bucket = min_posts_row[2]

        return bucket

    def min_num_of_posts(self, topic_id):
        min_posts_row = self._get_min_posts_row(topic_id)
        min_num_of_posts = int(min_posts_row[4])

        return min_num_of_posts

    def min_posts_num_of_authors(self, topic_id):
        min_posts_row = self._get_min_posts_row(topic_id)
        num_of_authors_when_min_posts = int(min_posts_row[3])

        return num_of_authors_when_min_posts

    def min_posts_post_author_ratio(self, topic_id):
        min_posts_row = self._get_min_posts_row(topic_id)
        post_author_ratio_when_min_posts = min_posts_row[5]

        return post_author_ratio_when_min_posts

        ##################################################################

    def _get_num_of_posts_bucket_by_given_number(self, topic_id, number):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        posts_series = df_group_by_topic["posts"]

        bucket = self._get_num_of_posts_bucket_by_fraction(df_group_by_topic, posts_series, number)
        return bucket

    def half_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 2)

    def third_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 3)

    def quarter_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 4)

    def fifth_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 5)

    def sixth_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 6)

    def seventh_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 7)

    def eighth_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 8)

    def ninth_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 9)

    def tenth_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 10)

    def one_part_eleven_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 11)

    def one_part_twelve_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 12)

    def one_part_thirteen_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 13)

    def one_part_fifteen_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 15)

    def one_part_twenty_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 20)

    def one_part_fifty_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 50)

    def one_part_one_hundred_num_of_posts_bucket(self, topic_id):
        return self._get_num_of_posts_bucket_by_given_number(topic_id, 100)

    def _get_opposite_num_of_posts_bucket_by_given_number(self, topic_id, number):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        df_group_by_topic = df_group_by_topic.sort_values(by='buckets', ascending=False)
        posts_series = df_group_by_topic["posts"]

        bucket = self._get_num_of_posts_bucket_by_fraction(df_group_by_topic, posts_series, number)
        return bucket

    def opposite_third_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 3)

    def opposite_quarter_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 4)

    def opposite_fifth_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 5)

    def opposite_sixth_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 6)

    def opposite_seventh_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 7)

    def opposite_eighth_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 8)

    def opposite_ninth_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 9)

    def opposite_tenth_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 10)

    def opposite_one_part_eleven_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 11)

    def opposite_one_part_twelve_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 12)

    def opposite_one_part_thirteen_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 13)

    def opposite_one_part_fifteen_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 15)

    def opposite_one_part_twenty_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 20)

    def opposite_one_part_fifty_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 50)

    def opposite_one_part_one_hundred_num_of_posts_bucket(self, topic_id):
        return self._get_opposite_num_of_posts_bucket_by_given_number(topic_id, 100)

    def _get_num_of_posts_bucket_by_fraction(self, df_group_by_topic, posts_series, number):
        sum = posts_series.sum()
        middle_num_of_posts = sum / number
        total_posts = 0
        for index, row in df_group_by_topic.iterrows():
            df_tuple = tuple(row)
            bucket = df_tuple[1]
            num_of_posts = df_tuple[3]
            total_posts += num_of_posts
            if total_posts >= middle_num_of_posts:
                return bucket

    def _get_num_of_authors_bucket_by_given_number(self, topic_id, number):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        authors_series = df_group_by_topic["authors"]

        bucket = self._get_num_of_authors_bucket_by_fraction(df_group_by_topic, authors_series, number)
        return bucket

    def half_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 2)

    def third_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 3)

    def quarter_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 4)

    def fifth_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 5)

    def sixth_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 6)

    def seventh_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 7)

    def eighth_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 8)

    def ninth_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 9)

    def tenth_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 10)

    def one_part_eleven_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 11)

    def one_part_twelve_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 12)

    def one_part_thirteen_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 13)

    def one_part_fifteen_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 15)

    def one_part_twenty_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 20)

    def one_part_fifty_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 50)

    def one_part_one_hundred_num_of_authors_bucket(self, topic_id):
        return self._get_num_of_authors_bucket_by_given_number(topic_id, 100)

    def _get_opposite_num_of_authors_bucket_by_given_number(self, topic_id, number):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        df_group_by_topic = df_group_by_topic.sort_values(by='buckets', ascending=False)

        authors_series = df_group_by_topic["authors"]

        bucket = self._get_num_of_authors_bucket_by_fraction(df_group_by_topic, authors_series, number)
        return bucket

    def opposite_third_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 3)

    def opposite_quarter_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 4)

    def opposite_fifth_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 5)

    def opposite_sixth_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 6)

    def opposite_seventh_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 7)

    def opposite_eighth_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 8)

    def opposite_ninth_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 9)

    def opposite_tenth_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 10)

    def opposite_one_part_eleven_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 11)

    def opposite_one_part_twelve_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 12)

    def opposite_one_part_thirteen_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 13)

    def opposite_one_part_fifteen_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 15)

    def opposite_one_part_twenty_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 20)

    def opposite_one_part_fifty_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 50)

    def opposite_one_part_one_hundred_num_of_authors_bucket(self, topic_id):
        return self._get_opposite_num_of_authors_bucket_by_given_number(topic_id, 100)

    def _get_num_of_authors_bucket_by_fraction(self, df_group_by_topic, authors_series, number):
        sum = authors_series.sum()
        middle_num_of_authors = sum / number
        total_authors = 0
        for index, row in df_group_by_topic.iterrows():
            df_tuple = tuple(row)
            bucket = df_tuple[1]
            num_of_authors = df_tuple[2]
            total_authors += num_of_authors
            if total_authors >= middle_num_of_authors:
                return bucket

    def average_num_of_posts(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        avg_posts = df_group_by_topic["posts"].mean()
        return avg_posts

    def average_num_of_authors(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        avg_authors = df_group_by_topic["authors"].mean()
        return avg_authors

    def median_num_of_posts(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        median_posts = df_group_by_topic["posts"].median()
        return median_posts

    def median_num_of_authors(self, topic_id):
        df_group_by_topic = self._topic_statistics_df[self._topic_statistics_df['topic_number'] == topic_id]
        median_authors = df_group_by_topic["authors"].median()
        return median_authors