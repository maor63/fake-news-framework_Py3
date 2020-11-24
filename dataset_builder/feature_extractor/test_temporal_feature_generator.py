from unittest import TestCase
from commons.commons import *
from datetime import datetime, timedelta
from DB.schema_definition import *
from dataset_builder.feature_extractor.temporal_feature_generator import TemporalFeatureGenerator


class TestTemporalFeatureGenerator(TestCase):
    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestTemporalFeatureGenerator, cls).setUpClass()
        cls._db = DB()
        cls._db.setUp()
        cls._posts = []
        cls._post_dictionary = {}

        cls._add_claim('c1', "2017-01-15 03:00:00")
        cls._add_author('a1', created_at='Sat Dec 19 12:13:12 +0000 2009')
        for i in range(4):
            hours = str(3 + i)
            cls._add_post('p{}'.format(i),
                          'content similar content one nice word Permutation Combination Python hello good bad',
                          "2017-01-16 0%s:00:00" % hours)
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))

        cls._add_author('a2', created_at='Sat Dec 19 12:13:12 +0000 2009')
        for i in range(4, 8):
            days = str(18 + i - 4)
            cls._add_post('p{}'.format(i),
                          'content similar content one nice word Permutation Combination hello good bad',
                          "2017-01-%s 05:00:00" % days)
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))
        cls._add_author('a3', False, 0, created_at='Sat Dec 26 12:13:12 +0000 2009')
        for i in range(8, 12):
            months = str(2 + i - 8)
            cls._add_post('p{}'.format(i),
                          'content similar content one nice word Permutation Combination Python hello good bad',
                          "2017-0%s-18 05:00:00" % months)
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))

        cls._add_claim('c2', "2017-06-15 03:00:00")
        cls._add_author('a4', created_at='Tue Dec 28 12:13:12 +0000 2009')
        for i in range(12, 16):
            cls._add_post('p{}'.format(i), 'content very different no relations', "2017-08-18 04:00:00")
            cls._add_claim_tweet_connection('c2', 'p{}'.format(i))

        cls._add_author('a5')
        for i in range(40, 44):
            cls._add_post('p{}'.format(i), 'content very different no relations https://www.google.com/',
                          "2017-06-16 03:00:00")
            cls._add_claim_tweet_connection('c2', 'p{}'.format(i))
        cls._add_author('a6')

        cls._db.add_posts_fast(cls._posts)
        cls._db.session.commit()
        cls._temporal_feature_generator = TemporalFeatureGenerator(cls._db, **{'authors': [], 'posts': {}})
        #

    def tearDown(self):
        self._db.session.close()
        pass

    # def test_stress_excution(self):
    #     claim_id = u'c10'
    #     self._add_claim(claim_id, "2017-01-15 03:00:00")
    #     self._add_author(u'a10', created_at=u'Sat Dec 19 12:13:12 +0000 2009')
    #     base_date = str_to_date("2017-01-16 00:00:00")
    #     for i in xrange(100, 1000):
    #         post_date = base_date + timedelta(hours=i)
    #         pots_id = u'p{}'.format(i)
    #         self._add_post(pots_id,
    #                        u'content similar content one nice word Permutation Combination Python hello good bad',
    #                        date_to_str(post_date))
    #         self._add_claim_tweet_connection(claim_id, pots_id)
    #
    #     claim_id = u'c20'
    #     self._add_claim(claim_id, "2017-01-15 03:00:00")
    #     self._add_author(u'a20', created_at=u'Sat Dec 19 12:13:12 +0000 2009')
    #     base_date = str_to_date("2017-01-16 00:00:00")
    #     for i in xrange(40000, 42000):
    #         post_date = base_date + timedelta(hours=i)
    #         pots_id = u'p{}'.format(i)
    #         self._add_post(pots_id,
    #                        u'content similar content one nice word Permutation Combination Python hello good bad',
    #                        date_to_str(post_date))
    #         self._add_claim_tweet_connection(claim_id, pots_id)
    #
    #     self._db.add_posts_fast(self._posts)
    #     self._db.session.commit()
    #     self._temporal_feature_generator.setUp()
    #     self._temporal_feature_generator.execute()

    def test_execute(self):
        self._temporal_feature_generator.execute()
        base_feature_name = 'TemporalFeatureGenerator'
        self._generic_test('c1', '%s_posts_temporal_sum_delta_time_30' % base_feature_name, 12)
        self._generic_test('c1', '%s_posts_temporal_sum_delta_time_365' % base_feature_name, 12)
        self._generic_test('c1', '%s_posts_temporal_sum_delta_time_1' % base_feature_name, 12)
        self._generic_test('c1', '%s_posts_temporal_std_delta_time_1' % base_feature_name, 0.430817603)
        self._generic_test('c1', '%s_posts_temporal_median_delta_time_1' % base_feature_name, 0)
        self._generic_test('c1', '%s_posts_temporal_mean_delta_time_1' % base_feature_name, 0.097560975)
        self._generic_test('c1', '%s_posts_temporal_max_delta_time_1' % base_feature_name, 4)
        self._generic_test('c1', '%s_posts_temporal_min_delta_time_1' % base_feature_name, 0)
        self._generic_test('c1', '%s_posts_temporal_kurtosis_delta_time_1' % base_feature_name, 52.9897271531887)
        self._generic_test('c1', '%s_posts_temporal_skew_delta_time_1' % base_feature_name, 6.62962719863894)

        self._generic_test('c1', '%s_authors_temporal_sum_delta_time_1' % base_feature_name, 3)
        self._generic_test('c1', '%s_authors_temporal_std_delta_time_1' % base_feature_name, 0.695970545353753)
        self._generic_test('c1', '%s_authors_temporal_median_delta_time_1' % base_feature_name, 0.0)
        self._generic_test('c1', '%s_authors_temporal_mean_delta_time_1' % base_feature_name, 0.375)
        self._generic_test('c1', '%s_authors_temporal_max_delta_time_1' % base_feature_name, 2)
        self._generic_test('c1', '%s_authors_temporal_min_delta_time_1' % base_feature_name, 0)
        self._generic_test('c1', '%s_authors_temporal_kurtosis_delta_time_1' % base_feature_name, 0.85952133194589)
        self._generic_test('c1', '%s_authors_temporal_skew_delta_time_1' % base_feature_name, 1.56430424345901)

        self._generic_test('c1', '%s_authors_temporal_sum_delta_time_30' % base_feature_name, 3)
        self._generic_test('c1', '%s_authors_temporal_std_delta_time_30' % base_feature_name, 0.0)
        self._generic_test('c1', '%s_authors_temporal_median_delta_time_30' % base_feature_name, 3.0)
        self._generic_test('c1', '%s_authors_temporal_mean_delta_time_30' % base_feature_name, 3.0)
        self._generic_test('c1', '%s_authors_temporal_max_delta_time_30' % base_feature_name, 3)
        self._generic_test('c1', '%s_authors_temporal_min_delta_time_30' % base_feature_name, 3)
        self._generic_test('c1', '%s_authors_temporal_kurtosis_delta_time_30' % base_feature_name, -3.0)
        self._generic_test('c1', '%s_authors_temporal_skew_delta_time_30' % base_feature_name, 0.0)

    def test_date_range_equal_intervals(self):
        publish_list = [p.date for p in self._posts[0:4]]
        self.assertListEqual(self._temporal_feature_generator.forward_date_range(publish_list, 1), [4])
        publish_list = [p.date for p in self._posts[4:8]]
        self.assertListEqual(self._temporal_feature_generator.forward_date_range(publish_list, 1), [1, 1, 1, 1])
        publish_list = [p.date for p in self._posts[8:12]]
        self.assertListEqual(self._temporal_feature_generator.forward_date_range(publish_list, 30), [2, 1, 1])
        publish_list = [p.date for p in self._posts[0:16]]
        self.assertListEqual(self._temporal_feature_generator.forward_date_range(publish_list, 30),
                             [8, 1, 1, 1, 1, 0, 0, 4])

    def _generic_test(self, author_guid, attribute, expected_value):
        db_val = self._db.get_author_feature(author_guid, attribute).attribute_value
        self.assertAlmostEqual(float(db_val), float(expected_value))

    @classmethod
    def _add_author(cls, author_guid, protected=True, verified=1, created_at='Mon Oct 12 12:40:21 +0000 2015'):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = author_guid
        author.author_screen_name = author_guid
        author.name = author_guid
        author.created_at = created_at
        author.domain = 'tests'
        author.statuses_count = 6
        author.followers_count = 10
        author.friends_count = 5
        author.favourites_count = 6
        author.listed_count = 6
        author.protected = protected
        author.verified = verified
        cls._db.addPost(author)
        cls._author = author

    @classmethod
    def _add_post(cls, post_id, content, date_str, domain='Microblog'):
        post = Post()
        post.author = cls._author.author_guid
        post.author_guid = cls._author.author_guid
        post.content = content
        post.title = post_id
        post.domain = domain
        post.post_id = post_id
        post.guid = post.post_id
        post.date = convert_str_to_unicode_datetime(date_str)
        post.created_at = post.date
        post.retweet_count = 4
        post.favorite_count = 3
        # cls._db.addPost(post)
        cls._posts.append(post)

        # self._author.statuses_count += 1

    @classmethod
    def _add_claim_tweet_connection(cls, claim_id, post_id):
        connection = Claim_Tweet_Connection()
        connection.claim_id = claim_id
        connection.post_id = post_id
        cls._db.addPost(connection)

    @classmethod
    def _add_claim(cls, claim_id, date_str):
        claim = Claim()
        claim.claim_id = claim_id
        claim.title = ''
        claim.domain = 'tests'
        claim.verdict_date = convert_str_to_unicode_datetime(date_str)
        cls._db.addPost(claim)
        pass
