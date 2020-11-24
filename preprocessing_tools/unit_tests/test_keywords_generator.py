from unittest import TestCase
from commons.commons import *
from DB.schema_definition import DB
from preprocessing_tools.fake_news_snopes_importer.fake_news_snopes_importer import FakeNewsSnopesImporter
from preprocessing_tools.keywords_generator import KeywordsGenerator


class TestKeywordsGenerator(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self.fake_news_snopes_importer = FakeNewsSnopesImporter(self._db)
        self.fake_news_snopes_importer._input_csv_file = 'data/input/FakeNewsSnopesImporter/Fake_News_Snopes_V3.csv'
        self.fake_news_snopes_importer.execute()

        self.keywords_generator = KeywordsGenerator(self._db)

    def tearDown(self):
        self._db.session.close()

    def test_generate_manual_keywords(self):
        self.keywords_generator.generate_manual_keywords()
        claim_keywords_connections = self._db.get_claim_keywords_connections_by_type('manual')
        self.assertEqual(claim_keywords_connections[0].keywords, 'Robert De Niro prostitution ring')
        self.assertEqual(claim_keywords_connections[4].keywords, 'Michelle Obama fire Roseanne Barr')
        self.assertEqual(claim_keywords_connections[15].keywords, 'Planned Parenthood Disney Princess abortion')
        self.assertEqual(claim_keywords_connections[25].keywords, 'Julia Roberts Celine Dion')
        pass

    def test_generate_random_keywords(self):
        self.keywords_generator.generate_random_keywords()
        claims = self._db.get_claims()
        claim_dict = {claim.claim_id: claim for claim in claims}
        claim_keywords_connections = self._db.get_claim_keywords_connections_by_type('random_5_keywords_1')

        self.assertRandomKeywordsCorrectly(claim_dict, claim_keywords_connections[0])
        self.assertRandomKeywordsCorrectly(claim_dict, claim_keywords_connections[4])
        self.assertRandomKeywordsCorrectly(claim_dict, claim_keywords_connections[15])
        self.assertRandomKeywordsCorrectly(claim_dict, claim_keywords_connections[25])
        self.assertRandomKeywordsCorrectly(claim_dict, claim_keywords_connections[16])

    def test_generate_post_tagging_keywords(self):
        self.keywords_generator.generate_post_tagging_keywords()
        claim_keywords_connections = self._db.get_claim_keywords_connections_by_type('pos_tagging_size_5')
        self.assertSetEqual(set(claim_keywords_connections[0].keywords.split()),
                            set('Robert De Niro prostitution ring'.lower().split()))
        self.assertSetEqual(set(claim_keywords_connections[4].keywords.split()),
                            set('Michelle firing Roseanne Star former'.lower().split()))
        self.assertSetEqual(set(claim_keywords_connections[15].keywords.split()),
                            set('tweet Parenthood Disney Princess abortion'.lower().split()))
        self.assertSetEqual(set(claim_keywords_connections[25].keywords.split()),
                            set('Julia Roberts Dion destroyed trump'.lower().split()))

    def assertRandomKeywordsCorrectly(self, claim_dict, connection):
        keywords = connection.keywords.split(' ')
        description = clean_claim_description(claim_dict[connection.claim_id].description, False)
        possible_key_words = set(description.split(' '))
        for keyword in keywords:
            self.assertIn(keyword, possible_key_words)
