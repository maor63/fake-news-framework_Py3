from unittest import TestCase

from DB.schema_definition import *
from preprocessing_tools.csv_to_table_importer import CsvToTableImporter
from preprocessing_tools.table_to_csv_exporter import TableToCsvExporter


class TestCsvToTableImporter(TestCase):
    def setUp(self):
        self._db = DB()

        self._db.setUp()
        self._posts = []
        self._author = None

    def tearDown(self):
        self._db.session.close()

    def test_import_tables_from_csv(self):
        self._table_to_csv_importer = CsvToTableImporter(self._db)
        self._table_to_csv_importer.setUp()
        self._table_to_csv_importer.execute()

        self._assert_table_rows_count(5, 'posts')
        self._assert_table_rows_count(2, 'authors')
        self._assert_table_rows_count(5, 'claim_tweet_connection')
        self._assert_table_rows_count(2, 'claims')

    # def test_double_import_no_overide(self):
    #     self._table_to_csv_importer = CsvToTableImporter(self._db)
    #     self._table_to_csv_importer.setUp()
    #     self._table_to_csv_importer.execute()
    #     self._table_to_csv_importer.execute()
    #     self._assert_table_rows_count(5, 'posts')
    #     self._assert_table_rows_count(2, 'authors')
    #     self._assert_table_rows_count(5, 'claim_tweet_connection')
    #     self._assert_table_rows_count(2, 'claims')


    def _assert_table_rows_count(self, rows_count, table_name):
        table = pd.read_sql_table(table_name, self._db.engine)
        self.assertEqual(len(table), rows_count)

    def _add_author(self, author_guid):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = 'test author'
        author.author_screen_name = author_guid
        author.name = 'test'
        author.domain = 'tests'
        author.statuses_count = 0
        author.created_at = "2017-06-14 05:00:00"
        self._db.add_author(author)
        self._author = author

    def _add_post(self, post_id, content, date_str="2017-06-14 05:00:00", domain='Microblog', post_type=None):
        post = Post()
        post.author = self._author.author_guid
        post.author_guid = self._author.author_guid
        post.content = content
        post.title = post_id
        post.domain = domain
        post.post_id = post_id
        post.guid = post.post_id
        post.date = convert_str_to_unicode_datetime(date_str)
        post.created_at = post.date
        post.post_type = post_type
        self._db.addPost(post)
        self._posts.append(post)


    def _get_params(self):
        posts = {self._author.author_guid: self._posts}
        params = params = {'authors': [self._author], 'posts': posts}
        return params

    def _add_claim_tweet_connection(self, claim_id, post_id):
        connection = Claim_Tweet_Connection()
        connection.claim_id = claim_id
        connection.post_id = post_id
        self._db.add_claim_connections([connection])
        pass
