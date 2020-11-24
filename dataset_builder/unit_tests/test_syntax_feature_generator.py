from unittest import TestCase

from DB.schema_definition import DB, Author, Post
from commons.commons import convert_str_to_unicode_datetime
from dataset_builder.feature_extractor.syntax_feature_generator import SyntaxFeatureGenerator


class TestSyntaxFeatureGenerator(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._posts = []
        self._author = None
        self.syntax_feature_generator = SyntaxFeatureGenerator(self._db, **{})


    def tearDown(self):
        self._db.session.close()
        pass

    def test_average_hashtags(self):
        self._add_author("author_guid")
        self._add_post("post1", "#content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 1", "2017-06-12 05:00:00")
        self._add_post("post3", "content 1 #tag #yes", "2017-06-12 05:00:00")
        self._add_post("post4", "content #1 #test #dont #fail #please", "2017-06-12 05:00:00")

        self.syntax_feature_generator.execute()

        author_feature = self._db.get_author_feature("author_guid", "SyntaxFeatureGenerator_average_hashtags")
        self.assertAlmostEqual(float(author_feature.attribute_value), 8.0 / 4, places=4)

    def test_average_links(self):
        self._add_author("author_guid")
        self._add_post("post1", "#content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "https://www.google.co.il 1", "2017-06-12 05:00:00")
        self._add_post("post3", "content 1 #tag http://www.google.co.il", "2017-06-12 05:00:00")
        self._add_post("post4", "http://www.bank.co.il #1 #test #dont http://www.ynet.co.il https://www.msn.co.il", "2017-06-12 05:00:00")

        self.syntax_feature_generator.execute()
        author_feature = self._db.get_author_feature("author_guid", "SyntaxFeatureGenerator_average_links")
        self.assertAlmostEqual(float(author_feature.attribute_value), 5.0 / 4, places=4)

    def test_average_user_mentions(self):
        self._add_author("author_guid")
        self._add_post("post1", "@content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 1", "2017-06-12 05:00:00")
        self._add_post("post3", "content 1 @tag #@es", "2017-06-12 05:00:00")
        self._add_post("post4", "content #1 @test @dont @fail #please", "2017-06-12 05:00:00")

        self.syntax_feature_generator.execute()

        author_feature = self._db.get_author_feature("author_guid", "SyntaxFeatureGenerator_average_user_mentions")
        self.assertAlmostEqual(float(author_feature.attribute_value), 6.0 / 4, places=4)

    def test_average_post_lenth(self):
        self._add_author("author_guid")
        self._add_post("post1", "content 1", "2017-06-12 05:00:00")
        self._add_post("post2", "content 1", "2017-06-12 05:00:00")
        self._add_post("post3", "content 1 @tag #@es", "2017-06-12 05:00:00")
        self._add_post("post4", "content #1 @test @dont @fail #please", "2017-06-12 05:00:00")

        self.syntax_feature_generator.execute()

        author_feature = self._db.get_author_feature("author_guid", "SyntaxFeatureGenerator_average_post_lenth")
        self.assertAlmostEqual(float(author_feature.attribute_value), 14.0 / 4, places=4)

    def _add_author(self, author_guid):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = 'test author'
        author.name = 'test'
        author.domain = 'tests'
        author.statuses_count = 0
        self._db.add_author(author)
        self._author = author

    def _add_post(self, title, content, date_str, domain='Microblog'):
        post = Post()
        post.author = self._author.author_guid
        post.author_guid = self._author.author_guid
        post.content = content
        post.title = title
        post.domain = domain
        post.post_id = title
        post.guid = post.post_id
        post.date = convert_str_to_unicode_datetime(date_str)
        post.created_at = post.date
        self._db.addPost(post)
        self._posts.append(post)

        self._author.statuses_count += 1
