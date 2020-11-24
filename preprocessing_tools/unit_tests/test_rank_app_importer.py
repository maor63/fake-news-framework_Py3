import re
from unittest import TestCase

from DB.schema_definition import DB
from preprocessing_tools.rank_app_importer import RankAppImporter


class TestRankAppImporter(TestCase):
    def setUp(self):
        self.normalize_actor_guid = "00f888bdfe92039ccbc440ab27b7804040f195e9dc367bc077270033"
        self.not_normalize_author_guid = "0cc3fd06f73d6613dec1e4e31bcd7c4efd430df3b00dd7fe092cfa5b"
        self._db = DB()
        self._db.setUp()

        self.rank_app_importer = RankAppImporter(self._db)
        self.rank_app_importer.setUp()

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()

    def test_read_from_folders(self):
        self.rank_app_importer._listdic = []
        self.rank_app_importer._read_apps_from_folder()
        result = self.rank_app_importer._listdic
        self.assertGreater(len(result), 0)
        self._db.session.close()

    def test_post_content_doesnt_contains_rank(self):
        self.rank_app_importer._listdic = []
        self.rank_app_importer._read_apps_from_folder()
        listdic = self.rank_app_importer._listdic
        posts_with_rank = []
        for post in listdic:
            if not re.match("[\d][.]0.", post['content']) is None:
                posts_with_rank.append(post['content'])
        self.assertEqual(len(posts_with_rank), 0)
        self._db.session.close()

    def test_only_posts_below_threshold_insert_to_db(self):
        self._db.setUp()
        self.rank_app_importer._rank_threshold = 2.0
        self.rank_app_importer.execute()
        posts = self._db.get_posts_by_author_guid(self.normalize_actor_guid)
        self.assertEqual(len(posts), 5)
        self._db.session.close()

    def test_not_normalized_posts_insert_correctly(self):
        self.rank_app_importer._rank_threshold = 5.0
        self.rank_app_importer.execute()
        posts = self._db.get_posts_by_author_guid(self.not_normalize_author_guid)
        self.assertEqual(len(posts), 16)
        self._db.session.close()
