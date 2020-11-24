from unittest import TestCase

from DB.schema_definition import DB, Post, Author, Claim_Tweet_Connection
from commons.commons import convert_str_to_unicode_datetime
from dataset_builder.feature_extractor.behavior_feature_generator import BehaviorFeatureGenerator
from dataset_builder.feature_extractor.feature_argument_parser import ArgumentParser


class TestBehaviorFeatureGenerator(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._posts = []
        self._author = None

    def tearDown(self):
        self._db.session.close()
        pass

    ######################## Average minute between posts tests ######################

    def test_average_minutes_between_posts_no_post_expected_0(self):
        self._add_author("author_guid")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._features = ['average_minutes_between_posts']
        self._behavior_feature_generator._targeted_fields = [
            {'source': {"table_name": "posts", "id": "author_guid", "target_field": "content",
                        "where_clauses": [{"field_name": 1, "value": 1}]},
             "connection": {},
             "destination": {}}
        ]
        result = self._behavior_feature_generator.average_minutes_between_posts(**{'posts': self._posts})
        self.assertEqual(0, result)

    def test_average_minutes_between_posts_one_post_expected_0(self):
        self._add_author("author_guid")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [
            {'source': {"table_name": "posts", "id": "author_guid", "target_field": "content",
                        "where_clauses": [{"field_name": 1, "value": 1}]},
             "connection": {},
             "destination": {}}
        ]
        self._behavior_feature_generator.execute()
        result_feature = self._db.get_author_feature("author_guid",
                                                     "BehaviorFeatureGenerator_average_minutes_between_posts")
        feature_value = getattr(result_feature, 'attribute_value')
        self.assertEqual('0', feature_value)

    def test_average_minutes_between_posts_3_post_expected_105(self):
        self._add_author("author_guid")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 2", "2017-06-12 06:00:00")
        self._add_post("post3", "content 3", "2017-06-12 08:30:00")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        result = self._behavior_feature_generator.average_minutes_between_posts(**{'posts': self._posts})
        self.assertEqual(105, result)

    ######################## Average posts per day tests ######################

    def test_average_posts_per_day_active_days_no_posts_expect_0(self):
        self._add_author("author_guid")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        result = self._behavior_feature_generator.average_posts_per_day_active_days(**{'posts': self._posts})
        self.assertEqual(0, result)

    def test_average_posts_per_day_1_active_days_1_post_each_expect_1(self):
        self._add_author("author_guid")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        result = self._behavior_feature_generator.average_posts_per_day_active_days(**{'posts': self._posts})
        self.assertAlmostEqual(1.0, result, 0.0000001)

    def test_average_posts_per_day_3_active_days_1_post_each_expect_1(self):
        self._add_author("author_guid")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "content 3", "2017-06-16 05:00:00")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        result = self._behavior_feature_generator.average_posts_per_day_active_days(**{'posts': self._posts})
        self.assertAlmostEqual(1.0, result, 0.0000001)

    def test_average_posts_per_day_3_active_days_1_first_2_secound_3_third_expect_2(self):
        self._add_author("author_guid")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "content 3", "2017-06-13 06:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5", "2017-06-16 04:00:00")
        self._add_post("post6", "content 6", "2017-06-16 05:00:00")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        result = self._behavior_feature_generator.average_posts_per_day_active_days(**{'posts': self._posts})
        self.assertEqual(2.0, result)

    def test_average_posts_per_day_3_active_days_1_first_2_secound_3_third_expect_2_represent_by_post(self):
        self._add_author("author_guid")
        self._add_post("post0", "the claim", "2017-06-10 05:00:00", "Claim")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "content 3", "2017-06-13 06:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5", "2017-06-16 04:00:00")
        self._add_post("post6", "content 6", "2017-06-16 05:00:00")
        self._add_post("post7", "content 7", "2017-06-16 06:00:00", "Not Microblog")
        self._add_claim_tweet_connection("post0", "post1")
        self._add_claim_tweet_connection("post0", "post2")
        self._add_claim_tweet_connection("post0", "post3")
        self._add_claim_tweet_connection("post0", "post4")
        self._add_claim_tweet_connection("post0", "post5")
        self._add_claim_tweet_connection("post0", "post6")
        self._add_claim_tweet_connection("post0", "post7")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [{'source': {'table_name': 'posts', 'id': 'post_id',
                                                                         "where_clauses": [{"field_name": "domain",
                                                                                            "value": "Claim"}]},
                                                              'connection': {'table_name': 'claim_tweet_connection',
                                                                             'source_id': 'claim_id',
                                                                             'target_id': 'post_id',
                                                                             "where_clauses": [{"val1": "source.date",
                                                                                                "val2": "dest.date",
                                                                                                "op": "<="}]},
                                                              'destination': {'table_name': 'posts', 'id': 'post_id',
                                                                              'target_field': 'content',
                                                                              "where_clauses": [{"field_name": "domain",
                                                                                                 "value": "Microblog"}]}}]
        self._behavior_feature_generator.execute()
        author_feature = self._db.get_author_feature('post0',
                                                     "BehaviorFeatureGenerator_average_posts_per_day_active_days")
        self.assertEqual('2.0', author_feature.attribute_value)
        author_feature = self._db.get_author_feature('post0',
                                                     "BehaviorFeatureGenerator_average_posts_per_day_total")
        self.assertGreater(float(author_feature.attribute_value), 0)

    def test_retweet_count_0_posts(self):
        self._add_author("author_guid")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [
            {'source': {'table_name': 'authors', 'id': 'author_guid', "target_field": "author_guid",
                        "where_clauses": [{"field_name": "1", "value": "1"}]},
             'connection': {},
             'destination': {}}]
        self._behavior_feature_generator.execute()
        author_feature = self._db.get_author_feature('author_guid',
                                                     "BehaviorFeatureGenerator_retweet_count")
        self.assertEqual('0', author_feature.attribute_value)

    def test_retweet_count_1_retweet(self):
        self._add_author("author_guid")
        self._add_post("post1", "RT @content 1", "2017-06-12 05:00:00")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [
            {'source': {'table_name': 'authors', 'id': 'author_guid', "target_field": "author_guid",
                        "where_clauses": [{"field_name": "1", "value": "1"}]},
             'connection': {},
             'destination': {}}]
        self._behavior_feature_generator.execute()
        author_feature = self._db.get_author_feature('author_guid',
                                                     "BehaviorFeatureGenerator_retweet_count")
        self.assertEqual('1', author_feature.attribute_value)

    def test_retweet_count_3_retweet(self):
        self._add_author("author_guid")
        self._add_post("post0", "the claim", "2017-06-10 05:00:00", "Claim")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "RT @content 3 RT @hi", "2017-06-13 06:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5 bla RT @bla", "2017-06-16 04:00:00")
        self._add_post("post6", "RT @content 6", "2017-06-16 05:00:00")
        self._add_post("post7", "content 7 RT @wow", "2017-06-16 06:00:00", "Not Microblog")
        self._add_claim_tweet_connection("post0", "post1")
        self._add_claim_tweet_connection("post0", "post2")
        self._add_claim_tweet_connection("post0", "post3")
        self._add_claim_tweet_connection("post0", "post4")
        self._add_claim_tweet_connection("post0", "post5")
        self._add_claim_tweet_connection("post0", "post6")
        self._add_claim_tweet_connection("post0", "post7")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [{'source': {'table_name': 'posts', 'id': 'post_id',
                                                                         "where_clauses": [{"field_name": "domain",
                                                                                            "value": "Claim"}]},
                                                              'connection': {'table_name': 'claim_tweet_connection',
                                                                             'source_id': 'claim_id',
                                                                             'target_id': 'post_id', },
                                                              'destination': {'table_name': 'posts', 'id': 'post_id',
                                                                              'target_field': 'content',
                                                                              "where_clauses": [{"field_name": "domain",
                                                                                                 "value": "Microblog"}]}}]
        self._behavior_feature_generator.execute()
        author_feature = self._db.get_author_feature('post0', "BehaviorFeatureGenerator_retweet_count")
        self.assertEqual('3', author_feature.attribute_value)

        author_feature = self._db.get_author_feature('post0', "BehaviorFeatureGenerator_average_retweets")
        self.assertEqual('0.5', author_feature.attribute_value)

    def test_received_retweets_count_0_retweets(self):
        self._add_author("author_guid")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [
            {'source': {'table_name': 'authors', 'id': 'author_guid', "target_field": "author_guid",
                        "where_clauses": [{"field_name": "1", "value": "1"}]},
             'connection': {},
             'destination': {}}]
        self._behavior_feature_generator.execute()
        author_feature = self._db.get_author_feature('author_guid', "BehaviorFeatureGenerator_retweet_count")
        self.assertEqual('0', author_feature.attribute_value)

    def test_received_retweets_count_1_retweets(self):
        self._add_author("author_guid")
        self._add_post("post1", "RT @author_guid content 1", "2017-06-12 05:00:00")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [
            {'source': {'table_name': 'authors', 'id': 'author_guid', "target_field": "author_guid",
                        "where_clauses": [{"field_name": "1", "value": "1"}]},
             'connection': {},
             'destination': {}}]
        self._behavior_feature_generator.execute()
        author_feature = self._db.get_author_feature('author_guid', "BehaviorFeatureGenerator_retweet_count")
        self.assertEqual('1', author_feature.attribute_value)

    def test_received_retweets_count_3_retweets_only_from_microblog_tweets(self):
        self._add_author("author_guid")
        self._add_post("post0", "the claim", "2017-06-10 05:00:00", "Claim")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "RT @author_guid content 3 RT @hi", "2017-06-13 06:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5 bla RT @author_guid", "2017-06-16 04:00:00")
        self._add_post("post6", "RT @content 6", "2017-06-16 05:00:00")
        self._add_post("post7", "content 7 RT @wow", "2017-06-16 06:00:00", "Not Microblog")
        self._add_claim_tweet_connection("post0", "post1")
        self._add_claim_tweet_connection("post0", "post2")
        self._add_claim_tweet_connection("post0", "post3")
        self._add_claim_tweet_connection("post0", "post4")
        self._add_claim_tweet_connection("post0", "post5")
        self._add_claim_tweet_connection("post0", "post6")
        self._add_claim_tweet_connection("post0", "post7")
        self._db.add_author(self._author)
        self._db.session.commit()
        params = self._get_params()
        self._behavior_feature_generator = BehaviorFeatureGenerator(self._db, **params)
        self._behavior_feature_generator._targeted_fields = [{'source': {'table_name': 'posts', 'id': 'post_id',
                                                                         "where_clauses": [{"field_name": "domain",
                                                                                            "value": "Claim"}]},
                                                              'connection': {'table_name': 'claim_tweet_connection',
                                                                             'source_id': 'claim_id',
                                                                             'target_id': 'post_id', },
                                                              'destination': {'table_name': 'posts', 'id': 'post_id',
                                                                              'target_field': 'content',
                                                                              "where_clauses": [{"field_name": "domain",
                                                                                                 "value": "Microblog"}]}}]
        self._behavior_feature_generator.execute()
        author_feature = self._db.get_author_feature('post0', "BehaviorFeatureGenerator_retweet_count")
        self.assertEqual('3', author_feature.attribute_value)

    ######################## argument_parser tests ######################

    def test_argument_parser_connection_conditions(self):
        self._add_author("author_guid")
        self._add_post("post0", "the claim", "2017-06-14 05:00:00", "Claim")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "content 3", "2017-06-13 06:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5", "2017-06-16 04:00:00")
        self._add_post("post6", "content 6", "2017-06-16 05:00:00")
        self._add_post("post7", "content 7", "2017-06-16 06:00:00", "Not Microblog")
        self._add_claim_tweet_connection("post0", "post1")
        self._add_claim_tweet_connection("post0", "post2")
        self._add_claim_tweet_connection("post0", "post3")
        self._add_claim_tweet_connection("post0", "post4")
        self._add_claim_tweet_connection("post0", "post5")
        self._add_claim_tweet_connection("post0", "post6")
        self._add_claim_tweet_connection("post0", "post7")
        self._db.add_author(self._author)
        self._db.session.commit()
        arg_parser = ArgumentParser(self._db)
        args = {'source': {'table_name': 'posts', 'id': 'post_id',
                           "where_clauses": [{"field_name": "domain",
                                              "value": "Claim"}]},
                'connection': {'table_name': 'claim_tweet_connection',
                               'source_id': 'claim_id',
                               'target_id': 'post_id',
                               "where_clauses": [{"val1": "source.date",
                                                  "val2": "dest.date",
                                                  "op": "<="}]},
                'destination': {'table_name': 'posts', 'id': 'post_id',
                                'target_field': 'content',
                                "where_clauses": [{"field_name": "domain",
                                                   "value": "Microblog"}]}}
        source_id_target_elements_dict = arg_parser._get_source_id_target_elements(args)
        actual = set([element.post_id for element in source_id_target_elements_dict["post0"]])
        expected = {'post4', 'post5', 'post6'}
        self.assertSetEqual(actual, expected)

    def test_argument_parser_connection_conditions_with_timedelta(self):
        self._add_author("author_guid")
        self._add_post("post0", "the claim", "2017-06-14 05:00:00", "Claim")
        self._add_post("post1", "content 1", "2017-06-12 06:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "content 3", "2017-06-13 06:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5", "2017-06-16 04:00:00")
        self._add_post("post6", "content 6", "2017-06-16 05:00:00")
        self._add_post("post7", "content 7", "2017-06-16 06:00:00", "Not Microblog")
        self._add_claim_tweet_connection("post0", "post1")
        self._add_claim_tweet_connection("post0", "post2")
        self._add_claim_tweet_connection("post0", "post3")
        self._add_claim_tweet_connection("post0", "post4")
        self._add_claim_tweet_connection("post0", "post5")
        self._add_claim_tweet_connection("post0", "post6")
        self._add_claim_tweet_connection("post0", "post7")
        self._db.add_author(self._author)
        self._db.session.commit()
        arg_parser = ArgumentParser(self._db)
        args = {'source': {'table_name': 'posts', 'id': 'post_id',
                           "where_clauses": [{"field_name": "domain",
                                              "value": "Claim"}]},
                'connection': {'table_name': 'claim_tweet_connection',
                               'source_id': 'claim_id',
                               'target_id': 'post_id',
                               "where_clauses": [{"val1": "source.date",
                                                  "val2": "dest.date",
                                                  "op": "timeinterval",
                                                  "delta": 1}]},
                'destination': {'table_name': 'posts', 'id': 'post_id',
                                'target_field': 'content',
                                "where_clauses": [{"field_name": "domain",
                                                   "value": "Microblog"}]}}
        source_id_target_elements_dict = arg_parser._get_source_id_target_elements(args)
        actual = set([element.post_id for element in source_id_target_elements_dict["post0"]])
        expected = {'post2', 'post3'}
        self.assertSetEqual(actual, expected)

    def test_argument_parser_connection_conditions_with_before_timedelta(self):
        self._add_author("author_guid")
        self._add_post("post0", "the claim", "2017-06-14 05:00:00", "Claim")
        self._add_post("post1", "content 1", "2017-06-12 06:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "content 3", "2017-06-13 06:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5", "2017-06-16 04:00:00")
        self._add_post("post6", "content 6", "2017-06-16 05:00:00")
        self._add_post("post7", "content 7", "2017-06-16 06:00:00", "Not Microblog")
        self._add_claim_tweet_connection("post0", "post1")
        self._add_claim_tweet_connection("post0", "post2")
        self._add_claim_tweet_connection("post0", "post3")
        self._add_claim_tweet_connection("post0", "post4")
        self._add_claim_tweet_connection("post0", "post5")
        self._add_claim_tweet_connection("post0", "post6")
        self._add_claim_tweet_connection("post0", "post7")
        self._db.add_author(self._author)
        self._db.session.commit()
        arg_parser = ArgumentParser(self._db)
        args = {'source': {'table_name': 'posts', 'id': 'post_id',
                           "where_clauses": [{"field_name": "domain",
                                              "value": "Claim"}]},
                'connection': {'table_name': 'claim_tweet_connection',
                               'source_id': 'claim_id',
                               'target_id': 'post_id',
                               "where_clauses": [{"val1": "source.date",
                                                  "val2": "dest.date",
                                                  "op": "before",
                                                  "delta": 1}]},
                'destination': {'table_name': 'posts', 'id': 'post_id',
                                'target_field': 'content',
                                "where_clauses": [{"field_name": "domain",
                                                   "value": "Microblog"}]}}
        source_id_target_elements_dict = arg_parser._get_source_id_target_elements(args)
        actual = set([element.post_id for element in source_id_target_elements_dict["post0"]])
        expected = {'post2', 'post3'}
        self.assertSetEqual(actual, expected)

    def test_argument_parser_connection_conditions_with_after_timedelta(self):
        self._add_author("author_guid")
        self._add_post("post0", "the claim", "2017-06-14 05:00:00", "Claim")
        self._add_post("post1", "content 1", "2017-06-12 06:00:00")
        self._add_post("post2", "content 2", "2017-06-13 05:00:00")
        self._add_post("post3", "content 3", "2017-06-15 05:00:00")
        self._add_post("post4", "content 4", "2017-06-16 03:00:00")
        self._add_post("post5", "content 5", "2017-06-16 04:00:00")
        self._add_post("post6", "content 6", "2017-06-16 05:00:00")
        self._add_post("post7", "content 7", "2017-06-16 06:00:00", "Not Microblog")
        self._add_claim_tweet_connection("post0", "post1")
        self._add_claim_tweet_connection("post0", "post2")
        self._add_claim_tweet_connection("post0", "post3")
        self._add_claim_tweet_connection("post0", "post4")
        self._add_claim_tweet_connection("post0", "post5")
        self._add_claim_tweet_connection("post0", "post6")
        self._add_claim_tweet_connection("post0", "post7")
        self._db.add_author(self._author)
        self._db.session.commit()
        arg_parser = ArgumentParser(self._db)
        args = {'source': {'table_name': 'posts', 'id': 'post_id',
                           "where_clauses": [{"field_name": "domain",
                                              "value": "Claim"}]},
                'connection': {'table_name': 'claim_tweet_connection',
                               'source_id': 'claim_id',
                               'target_id': 'post_id',
                               "where_clauses": [{"val1": "source.date",
                                                  "val2": "dest.date",
                                                  "op": "after",
                                                  "delta": 1}]},
                'destination': {'table_name': 'posts', 'id': 'post_id',
                                'target_field': 'content',
                                "where_clauses": [{"field_name": "domain",
                                                   "value": "Microblog"}]}}
        source_id_target_elements_dict = arg_parser._get_source_id_target_elements(args)
        actual = set([element.post_id for element in source_id_target_elements_dict["post0"]])
        expected = {'post3'}
        self.assertSetEqual(actual, expected)

    def _add_author(self, author_guid):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = 'test author'
        author.author_screen_name = author_guid
        author.name = 'test'
        author.domain = 'tests'
        author.statuses_count = 0
        author.created_at = "2017-06-14 05:00:00"
        # self._db.add_author(author)
        self._author = author

    def _add_post(self, title, content, date_str, domain='Microblog'):
        post = Post()
        post.author = self._author.author_guid
        post.author_guid = self._author.author_guid
        post.content = content
        post.title = title
        post.domain = domain
        post.post_id = title
        post.guid = post.post_id
        post.date = convert_str_to_unicode_datetime(date_str)
        post.created_at = post.date
        self._db.addPost(post)
        self._posts.append(post)

        self._author.statuses_count += 1

    def _get_params(self):
        posts = {self._author.author_guid: self._posts}
        params = {'authors': [self._author], 'posts': posts}
        return params

    def _add_claim_tweet_connection(self, claim_id, post_id):
        connection = Claim_Tweet_Connection()
        connection.claim_id = claim_id
        connection.post_id = post_id
        self._db.add_claim_connections([connection])
        pass
