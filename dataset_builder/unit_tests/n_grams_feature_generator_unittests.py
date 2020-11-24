# Written by Lior Bass 3/29/2018
import unittest

from DB.schema_definition import DB, Post, Author, date
from configuration.config_class import getConfig
from dataset_builder.feature_extractor.n_gram_feature_generator import N_Grams_Feature_Generator


class N_Grams_Feature_Generator_Unittests(unittest.TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._domain = 'test'
        self._posts = []
        self._authors = []

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()

    def execute_module(self):
        authors = self._authors
        posts = self._db.get_posts_by_domain('test')
        parameters = {"authors": authors, "posts": posts, "graphs": []}
        n_gram_module = N_Grams_Feature_Generator(self._db, **parameters)
        n_gram_module._stemming = False
        n_gram_module.execute(window_start=None)

    def test_simple_case(self):
        self._add_author('1')
        self._add_post('do that',
                       'This article includes a list of references, but its sources remain unclear because it has insufficient inline citations. Please help to improve this article by introducing more precise citations.',
                       '1')
        self._add_post('to do ', 'article citations insufficient inline because the damn thing will not do that', '1')
        self._add_author('2')
        self._add_post('this was a triumph', 'im making a note here insufficient inline', '2')
        self.execute_module()
        features = self._db.get_author_features()
        db_val = self._db.get_author_feature('1', "2_gram_insufficient_inline").attribute_value
        self.assertEqual(db_val, str(2))

    def _add_author(self, author_guid):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = 'test author'
        author.name = 'test'
        author.domain = 'test'
        self._db.add_author(author)
        self._db.session.commit()
        self._authors.append(author)

    def _add_post(self, title, content, author_guid):
        post = Post()
        post.author = author_guid
        post.author_guid = author_guid
        post.content = content
        post.title = title
        post.domain = 'test'
        post.post_id = len(self._posts)
        post.guid = post.post_id
        post.date = date('2020-01-01 23:59:59')
        self._db.addPost(post)
        self._db.session.commit()
        self._posts.append(post)
