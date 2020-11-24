from unittest import TestCase

from DB.schema_definition import DB, Author, Post
from commons.commons import convert_str_to_unicode_datetime
from dataset_builder.feature_extractor.account_properties_feature_generator import AccountPropertiesFeatureGenerator
from dateutil import parser
import datetime

class TestAccountPropertiesFeatureGenerator(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self.author_guid = "author_guid"

        author = Author()
        author.author_guid = self.author_guid
        author.author_full_name = 'author'
        author.name = 'author_name'
        author.author_screen_name = 'author_screen_name'
        author.domain = 'Microblog'
        author.statuses_count = 10
        author.friends_count = 5
        author.followers_count = 6
        author.favourites_count = 8
        author.author_sub_type = "bot"
        author.author_type = "bad"
        author.created_at = "2017-06-17 05:00:00"
        author.default_profile = True
        author.default_profile_image = True
        author.verified = True
        self._db.add_author(author)

        post = Post()
        post.author = self.author_guid
        post.author_guid = self.author_guid
        post.content = "content"
        post.title = "title"
        post.domain = "domain"
        post.post_id = "post_id"
        post.guid = post.post_id
        post.date = convert_str_to_unicode_datetime("2017-06-14 05:00:00")
        post.created_at = post.date
        self._db.addPost(post)


        self._db.session.commit()
        self.feature_prefix = "AccountPropertiesFeatureGenerator_"
        self.account_properties_feature_generator = AccountPropertiesFeatureGenerator(self._db, **{'authors': [author], 'posts': {self.author_guid: [post]}})
        self.account_properties_feature_generator.execute()

    def tearDown(self):
        self._db.session.close()
        pass

    def test_account_age(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "account_age")
        account_creation_date = parser.parse("2017-06-17 05:00:00").date()
        today_date = datetime.date.today()
        delta = today_date - account_creation_date
        self.assertEqual(delta.days, int(author_feature.attribute_value))

    def test_number_followers(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "number_followers")
        self.assertEqual(6, int(author_feature.attribute_value))

    def test_number_friends(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "number_friends")
        self.assertEqual(5, int(author_feature.attribute_value))

    def test_friends_followers_ratio(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "friends_followers_ratio")
        self.assertAlmostEqual(5.0 / 6, float(author_feature.attribute_value), places=5)

    def test_number_of_crawled_posts(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "number_of_crawled_posts")
        self.assertEqual("1", author_feature.attribute_value)

    def test_number_of_posts(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "number_of_posts")
        self.assertEqual(10, int(author_feature.attribute_value))

    def test_default_profile(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "default_profile")
        self.assertEqual("1", author_feature.attribute_value)

    def test_default_profile_image(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "default_profile_image")
        self.assertEqual("1", author_feature.attribute_value)

    def test_verified(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "verified")
        self.assertEqual('1', author_feature.attribute_value)

    def test_screen_name_length(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "screen_name_length")
        self.assertEqual(18, int(author_feature.attribute_value))

    def test_author_screen_name(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "author_screen_name")
        self.assertEqual("author_screen_name", author_feature.attribute_value)

    def test_author_type(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "author_type")
        self.assertEqual("bad", author_feature.attribute_value)

    def test_author_sub_type(self):
        author_feature = self._db.get_author_feature(self.author_guid, self.feature_prefix + "author_sub_type")
        self.assertEqual("bot", author_feature.attribute_value)
