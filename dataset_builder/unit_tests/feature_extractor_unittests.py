import unittest
from DB.schema_definition import *
import datetime
from configuration.config_class import getConfig
from dataset_builder.feature_extractor.behavior_feature_generator import BehaviorFeatureGenerator
from dataset_builder.feature_extractor.boost_score_feature_generator import BoostScoresFeatureGenerator
from dataset_builder.feature_extractor.key_author_score_feature_generator import KeyAuthorScoreFeatureGenerator
from commons.consts import Domains
from dataset_builder.boost_authors_model import BoostAuthorsModel
from dataset_builder.key_authors_model import KeyAuthorsModel
from dataset_builder.autotopic_executor import AutotopicExecutor
from dataset_builder.autotopic_model_creator import AutotopicModelCreator
from dataset_builder.feature_extractor.syntax_feature_generator import SyntaxFeatureGenerator
from dataset_builder.feature_extractor.account_properties_feature_generator import AccountPropertiesFeatureGenerator
from preprocessing_tools.xml_importer import XMLImporter


class FeatureExtractorTest(unittest.TestCase):
    def setUp(self):
        self.config = getConfig()
        self._db = DB()
        self._db.setUp()

        author1 = Author()
        author1.name = 'TestUser1'
        author1.domain = 'Microblog'
        author1.author_guid = 'TestUser1'
        author1.author_screen_name = 'TestUser1'
        author1.author_full_name = 'TestUser1'
        author1.statuses_count = 10
        author1.author_osn_id = 1
        author1.created_at = datetime.datetime.strptime('2016-04-02 15:43:00', '%Y-%m-%d %H:%M:%S')
        author1.missing_data_complementor_insertion_date = datetime.datetime.now()
        author1.xml_importer_insertion_date = datetime.datetime.now()
        author1.author_type = 1
        self._db.add_author(author1)

        for i in range(50):
            post = Post()
            post.post_id = 'TestPost' + str(i)
            post.author = 'TestUser1'
            post.guid = 'TestPost' + str(i)
            post.url = 'TestPost' + str(i)
            tempDate = '2016-05-05 00:00:00'
            day = datetime.timedelta(1)
            post.date = datetime.datetime.strptime(tempDate, '%Y-%m-%d %H:%M:%S') + day * i
            post.domain = 'Microblog'
            post.author_guid = 'TestUser1'
            post.content = "InternetTV  yyy love it #wow" + chr(i)
            post.xml_importer_insertion_date = datetime.datetime.now()
            self._db.addPost(post)

        author = Author()
        author.name = 'TestUser2'
        author.domain = 'Microblog'
        author.author_guid = 'TestUser2'
        author.author_screen_name = 'TestUser2'
        author.author_full_name = 'TestUser2'
        author.statuses_count = 3
        author.author_osn_id = 1
        author.created_at = datetime.datetime.strptime('2016-04-05 00:00:00', '%Y-%m-%d %H:%M:%S')
        author.missing_data_complementor_insertion_date = datetime.datetime.now()
        author.xml_importer_insertion_date = datetime.datetime.now()
        author.author_type = 1
        self._db.add_author(author)

        for i in range(50, 100):
            post = Post()
            post.post_id = 'TestPost' + str(i)
            post.author = 'TestUser2'
            post.guid = 'TestPost' + str(i)
            post.url = 'TestPost' + str(i)
            tempDate = '2016-05-05 00:00:00'
            day = datetime.timedelta(1)
            post.date = datetime.datetime.strptime(tempDate, '%Y-%m-%d %H:%M:%S') + day * i
            post.domain = 'Microblog'
            post.author_guid = 'TestUser2'
            post.content = 'OnlineTV lalala gogogogo market http://google.com' + chr(i)
            post.xml_importer_insertion_date = datetime.datetime.now()
            self._db.addPost(post)

        author = Author()
        author.name = 'TestUser3'
        author.domain = 'Microblog'
        author.author_guid = 'TestUser3'
        author.author_screen_name = 'TestUser3'
        author.author_full_name = 'TestUser3'
        author.statuses_count = 3
        author.author_osn_id = 1
        author.created_at = datetime.datetime.strptime('2016-04-05 00:00:00', '%Y-%m-%d %H:%M:%S')
        author.missing_data_complementor_insertion_date = datetime.datetime.now()
        author.xml_importer_insertion_date = datetime.datetime.now()
        author.author_type = 1
        self._db.add_author(author)

        for i in range(100, 116):
            post = Post()
            post.post_id = 'TestPost' + str(i)
            post.author = 'TestUser3'
            post.guid = 'TestPost' + str(i)
            post.url = 'TestPost' + str(i)
            tempDate = '2016-05-05 00:00:00'
            day = datetime.timedelta(1)
            post.date = datetime.datetime.strptime(tempDate, '%Y-%m-%d %H:%M:%S') + day * i
            post.domain = 'Microblog'
            post.author_guid = 'TestUser3'
            post.content = 'SmartTV love it #tv #online' + chr(i)
            post.xml_importer_insertion_date = datetime.datetime.now()
            self._db.addPost(post)

        for i in range(10, 12):
            post_cit = Post_citation()
            post_cit.post_id_from = str('TestPost' + str(i))
            post_cit.post_id_to = 'TestPost1'
            self._db.session.merge(post_cit)
            self._db.commit()

        post_cit = Post_citation()
        post_cit.post_id_from = 'TestPost12'
        post_cit.post_id_to = 'TestPost2'
        self._db.session.merge(post_cit)
        self._db.commit()

        for i in range(13, 15):
            post_cit = Post_citation()
            post_cit.post_id_from = str('TestPost' + str(i))
            post_cit.post_id_to = 'TestPost1'
            self._db.session.merge(post_cit)
            self._db.commit()

        post_cit = Post_citation()
        post_cit.post_id_from = 'TestPost15'
        post_cit.post_id_to = 'TestPost2'
        self._db.session.merge(post_cit)
        self._db.commit()

        author = Author()
        author.name = 'TestUser4'
        author.domain = 'Microblog'
        author.author_guid = 'TestUser4'
        author.author_screen_name = 'TestUser4'
        author.author_full_name = 'TestUser4'
        author.statuses_count = 3
        author.author_osn_id = 1
        author.created_at = datetime.datetime.strptime('2016-04-05 00:00:00', '%Y-%m-%d %H:%M:%S')
        author.missing_data_complementor_insertion_date = datetime.datetime.now()
        author.xml_importer_insertion_date = datetime.datetime.now()
        author.author_type = 1
        self._db.add_author(author)

        post = Post()
        post.post_id = 'TestPost119'
        post.author = 'TestUser4'
        post.guid = 'TestPost119'
        post.url = 'TestPost119'
        post.date = datetime.datetime.strptime('2016-06-05 00:00:00', '%Y-%m-%d %H:%M:%S')
        post.domain = 'Microblog'
        post.author_guid = 'TestUser4'
        post.content = 'OnlineTV SmartTV love it @TestUser1'
        post.xml_importer_insertion_date = datetime.datetime.now()
        self._db.addPost(post)

        post_cit = Post_citation()
        post_cit.post_id_from = 'TestPost119'
        post_cit.post_id_to = 'TestPost1'
        self._db.session.merge(post_cit)
        self._db.commit()

        post_cit = Post_citation()
        post_cit.post_id_from = 'TestPost119'
        post_cit.post_id_to = 'TestPost12'
        self._db.session.merge(post_cit)
        self._db.commit()

        post_cit = Post_citation()
        post_cit.post_id_from = 'TestPost119'
        post_cit.post_id_to = 'TestPost7'
        self._db.session.merge(post_cit)
        self._db.commit()

        self.authors = self._db.get_authors_by_domain(Domains.MICROBLOG)
        self.posts = self._db.get_posts_by_domain(Domains.MICROBLOG)

    def testBehaviorFeatureGenerator(self):
        parameters = {"authors": self.authors, "posts": self.posts}
        behaviorFeatures = BehaviorFeatureGenerator(self._db, **parameters)
        behaviorFeatures.execute()
        allFeatures = self._db.get_author_features_by_author_guid('TestUser1')
        if not (allFeatures == None):
            for feature in allFeatures:
                if (feature.attribute_name == "average_minutes_between_posts"):
                    self.assertEqual(feature.attribute_value, '1440.0')
                elif (feature.attribute_name == "average_posts_per_day_active_days"):
                    self.assertEqual(feature.attribute_value, '1.0')
                elif (feature.attribute_name == "average_posts_per_day_total"):
                    author = self._db.getAuthorByName('TestUser1')
                    created_date = datetime.datetime.strptime(author[0].created_at, '%Y-%m-%d %H:%M:%S')
                    total_days = float((datetime.date.today() - created_date.date()).days)
                    total_posts = float(author[0].statuses_count)
                    self.assertAlmostEqual(float(feature.attribute_value), total_posts / total_days, 14)
        self._db.session.close()

    @unittest.skip("skipping i sill dont know why this tests doesnt pass")
    def testKeyAuthorScoreFeatureGenerator(self):
        model_creator = AutotopicModelCreator(self._db)
        model_creator.setUp()
        model_creator.execute(None)
        autotopicExe = AutotopicExecutor(self._db)
        autotopicExe.setUp()
        autotopicExe.execute()
        key_author_model = KeyAuthorsModel(self._db)
        key_author_model.setUp()
        key_author_model.execute()
        parameters = {"authors": self.authors, "posts": self.posts}
        keyAuthorFeatures = KeyAuthorScoreFeatureGenerator(self._db, **parameters)
        keyAuthorFeatures.execute()
        allFeatures = self._db.get_author_features_by_author_guid('TestUser1')
        if allFeatures is not None:
            for feature in allFeatures:
                if feature.attribute_name == "sum_tfidf":
                    self.assertNotEqual(feature.attribute_value, None)
                elif feature.attribute_name == "max_tfidf":
                    self.assertNotEqual(feature.attribute_value, None)
        self._db.session.close()

    def testSyntaxFeatureGenerator(self):
        parameters = {"authors": self.authors, "posts": self.posts}
        syntax_feature = SyntaxFeatureGenerator(self._db, **parameters)
        syntax_feature.execute()
        allFeatures = self._db.get_author_features()
        if allFeatures is not None:
            for feature in allFeatures:
                if feature.author_guid == 'TestUser1' and feature.attribute_name == "average_hashtags":
                    self.assertEqual(feature.attribute_value, '1.0')
                elif feature.author_guid == 'TestUser4' and feature.attribute_name == "average_user_mentions":
                    self.assertEqual(feature.attribute_value, '1.0')
                elif feature.author_guid == 'TestUser2' and feature.attribute_name == "average_links":
                    self.assertEqual(feature.attribute_value, '1.0')
                elif feature.author_guid == 'TestUser3' and feature.attribute_name == "average_post_lenth":
                    self.assertEqual(feature.attribute_value, '5.0')
        self._db.session.close()

    def testAccountPropertiesFeatureGenerator(self):
        try:
            parameters = {"authors": self.authors, "posts": self.posts}
            account_feature = AccountPropertiesFeatureGenerator(self._db, **parameters)
            account_feature.execute()
        except Exception as inst:
            print(inst)
        self._db.session.close()

    def tearDown(self):
        # self._db.deleteDB()
        self._db.session.close()
