from unittest import TestCase

from DB.schema_definition import DB
from preprocessing_tools.comlex_claim_tweet_importer.ComLexClaimTweetImporter import ComLexClaimTweetImporter


class TestComlexClaimTweetImporter(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()

        self._comlex_claim_tweet_importer = ComLexClaimTweetImporter(self._db)

    def tearDown(self):
        self._db.session.close()
        # self._db.deleteDB()

    def test__import_claims(self):
        self._comlex_claim_tweet_importer.execute(None)
        claims = self._db.get_claims()
        self.assertEqual(2296, len(claims))
