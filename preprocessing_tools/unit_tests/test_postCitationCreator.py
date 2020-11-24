from unittest import TestCase

import datetime

from DB.schema_definition import DB, Post
from commons.commons import compute_author_guid_by_author_name
from preprocessing_tools.post_citation_creator import PostCitationCreator


class TestPostCitationCreator(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()

    def test_execute(self):
        _author_guid1 = '05cd2e04ffaf3c5dabd03d13b63afab6'
        post_content = "InternetTV love it #wow https://t.co/tRRt https://t.co/Ao1KOOx77H"
        # "https://twitter.com/AmichaiStein1/status/725022086377431041/photo/1"
        self.create_post(_author_guid1, post_content)

        pc = PostCitationCreator(self._db)
        pc.execute()

        posts = self._db.get_posts_by_domain('Microblog')
        self.assertEqual(0, len(posts))

    def create_post(self, _author_guid1, post_content):
        post = Post()
        post.post_id = 'TestPost'
        post.author = 'TechmarketNG'
        post.guid = 'TestPost'
        post.url = 'Url_From'
        tempDate = '2016-05-05 00:00:00'
        day = datetime.timedelta(1)
        post.date = datetime.datetime.strptime(tempDate, '%Y-%m-%d %H:%M:%S') + day
        post.domain = 'Microblog'
        post.author_guid = _author_guid1
        post.content = post_content
        post.xml_importer_insertion_date = datetime.datetime.now()
        self._db.addPost(post)
        self._db.commit()
