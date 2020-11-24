from unittest import TestCase

from DB.schema_definition import DB
from preprocessing_tools.asonam_honeypot_importer.asonam_honeypot_importer import AsonamHoneypotImporter


class TestAsonamHoneypotImporter(TestCase):
    def setUp(self):
        self.db = DB()
        self.db.setUp()
        self.honeypot_importer = AsonamHoneypotImporter(self.db)

    def test_load_data_file(self):
        type_to_users_twits = self.honeypot_importer.load_file('data/input/Asonam_test_data/honeypot_test_data.txt')
        self.assertEqual(len(type_to_users_twits['human']), 3)
        self.assertEqual(len(type_to_users_twits['bot']), 3)

    def test_get_tweet_from_tweets_id(self):
        tweets = self.honeypot_importer.get_tweets([580241798288801792,580238740498432000,580235372128501761,580234434324094976,580232128299986945,580224280664784896,580223262497591296,580220596228849664])
        self.assertEqual(len(tweets), 8)
        pass

    # def test_execute(self):
    #     self.honeypot_importer.execute()


