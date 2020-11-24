#
# Created by Aviad on 04-Nov-16 10:22 PM.
#

import unittest
from DB.unit_tests.test_base import TestBase
from configuration.config_class import getConfig
from DB.schema_definition import DB, Post_citation
from preprocessing_tools.tumblr_importer.tumblr_importer import TumblrImporter
from commons.consts import Domains
from dataset_builder.feature_extractor.boost_score_feature_generator import BoostScoresFeatureGenerator
from dataset_builder.boost_authors_model import BoostAuthorsModel



class TestTumblrImporterBoostScoreFeatureGenerator(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.config = getConfig()
        self._start_date = self.config.eval("DEFAULT", "start_date")
        #self._end_date = self.config.get("DEFAULT", "end_date")

        self._tsv_files_path = self.config.get("TumblrImporter", "tsv_test_files_boost_score_feature_generator")
        self._db = DB()
        self._db.setUp()

        self._tumblr_parser = TumblrImporter(self._db)
        self._tumblr_parser.setUp(self._tsv_files_path)
        self._tumblr_parser.execute()

        self._author_guid = "f0f4bb42-3fed-322a-b71a-681179d47ea1"
        self._original_post_id = "130277126878"
        self._original_post_url = "http://tmblr.co/ZBqKGn1vL7pBS1"

        self._non_origin_post_id1 = "130277126879"
        self._non_origin_post_url1 = "http://tmblr.co/ZBqKGn1vL7pBS2"

        self._non_origin_post_id2 = "130277126880"
        self._non_origin_post_url2 = "http://tmblr.co/ZBqKGn1vL7pBS3"

        self._non_origin_post_id3 = "130277126881"
        self._non_origin_post_url3 = "http://tmblr.co/ZBqKGn1vL7pBS4"

        self._tumblr_parser.setUp(self._tsv_files_path)
        self._tumblr_parser.execute()

        self._create_post_citations()

        boost_score_model = BoostAuthorsModel(self._db)
        boost_score_model.execute(self._start_date)


        authors = self._db.get_authors_by_domain(Domains.MICROBLOG)
        posts = self._db.get_posts_by_domain(Domains.MICROBLOG)
        parameters = {"authors": authors, "posts": posts}

        self._boost_score_feature_generator = BoostScoresFeatureGenerator(self._db, **parameters)
        self._boost_score_feature_generator.execute()

        self._author_features = self._db.get_author_features_by_author_guid(author_guid=self._author_guid)
        self._author_features_dict = self._create_author_features_dictionary(self._author_features)

    def test_boost_score_avg(self):

        attribute_value = self._author_features_dict["boost_score_avg"]
        attribute_value = float(attribute_value)
        self.assertEqual(0.00015625, attribute_value)


    def test_boost_score_sum(self):

        attribute_value = self._author_features_dict["boost_score_sum"]
        attribute_value = float(attribute_value)
        self.assertEqual(0.00015625, attribute_value)

    def test_boost_score_std_dev(self):

        attribute_value = self._author_features_dict["boost_score_std_dev"]
        attribute_value = float(attribute_value)
        self.assertEqual(0.0, attribute_value)

    def test_boosting_timeslots_count(self):

        attribute_value = self._author_features_dict["boosting_timeslots_count"]
        attribute_value = float(attribute_value)
        self.assertEqual(0.0, attribute_value)

    def test_count_authors_sharing_boosted_posts(self):

        attribute_value = self._author_features_dict["count_authors_sharing_boosted_posts"]
        attribute_value = float(attribute_value)
        self.assertEqual(0.0, attribute_value)

    def tearDown(self):
        self._db.deleteDB()
        pass

    def _create_author_features_dictionary(self, author_features):
        author_features_dict = {}
        for author_feature in author_features:
            attribute_name = author_feature.attribute_name
            attribute_value = author_feature.attribute_value
            author_features_dict[attribute_name] = attribute_value

        return author_features_dict

    def _create_post_citations(self):
        post_citations = []

        post_citation1 = Post_citation(post_id_from=self._non_origin_post_id1, post_id_to=self._original_post_id,
                                      url_from=self._non_origin_post_url1, url_to=self._original_post_url)

        post_citations.append(post_citation1)

        post_citation2 = Post_citation(post_id_from=self._non_origin_post_id2, post_id_to=self._original_post_id,
                                       url_from=self._non_origin_post_url2, url_to=self._original_post_url)
        post_citations.append(post_citation2)

        post_citation3 = Post_citation(post_id_from=self._non_origin_post_id3, post_id_to=self._non_origin_post_id1,
                                       url_from=self._non_origin_post_url3, url_to=self._non_origin_post_url1)
        post_citations.append(post_citation3)

        self._db.addReferences(post_citations)
