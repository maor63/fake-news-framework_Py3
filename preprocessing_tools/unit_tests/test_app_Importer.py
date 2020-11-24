from unittest import TestCase

from DB.schema_definition import DB
from preprocessing_tools.app_importer import AppImporter


class TestAppImporter(TestCase):
    def setUp(self):
        self.bad_actor_guid = "e2f8a58933d5e673d9c673c442cea1b73e9732d27a0f13472fde19f0"
        self.good_actor_guid = "0a2f4a19fb5066c3a67fc9b3325515b8bf0db66b7fec92b63da564a9"

        self._db = DB()
        self._db.setUp()

        self.app_importer = AppImporter(self._db)
        self.app_importer.setUp()
        self._domain = self.app_importer._domain
        self.app_importer.execute()

    def test_read_from_folders(self):
        self.app_importer._listdic = []
        self.app_importer._read_apps_from_folder()
        result = self.app_importer._listdic
        self.assertGreater(len(result), 0)

    def test_good_actor_posts_inserted_to_db(self):
        posts = self._db.get_posts_by_author_guid(self.good_actor_guid)
        self.assertGreater(len(posts), 0)

    def test_bad_actor_posts_inserted_to_db(self):
        posts = self._db.get_posts_by_author_guid(self.bad_actor_guid)
        self.assertGreater(len(posts), 0)

    def test_good_actor_inserted_to_db(self):
        self.app_importer.execute()
        author = self._db.get_author_by_author_guid(self.good_actor_guid)
        self.assertEqual(author.author_type, "good_actor")

    def test_bad_actor_inserted_to_db(self):
        self.app_importer.execute()
        author = self._db.get_author_by_author_guid(self.bad_actor_guid)
        self.assertEqual(author.author_type, "bad_actor")

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()