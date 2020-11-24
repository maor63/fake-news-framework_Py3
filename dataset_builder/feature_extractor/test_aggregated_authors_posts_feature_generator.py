from unittest import TestCase

from DB.schema_definition import *
from dataset_builder.feature_extractor.aggregated_authors_posts_feature_generator import \
    AggregatedAuthorsPostsFeatureGenerator


class TestAggregatedAuthorsPostsFeatureGenerator(TestCase):
    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestAggregatedAuthorsPostsFeatureGenerator, cls).setUpClass()
        cls._db = DB()
        cls._db.setUp()
        cls._posts = []
        cls._post_dictionary = {}
        cls._author = None

        cls._add_claim('c1', "2017-06-15 03:00:00")
        cls._add_author('a1_med', created_at='Sat Dec 19 12:13:12 +0000 2009')
        for i in range(4, 8):
            cls._add_post('p{}'.format(i), 'content', "2017-06-18 05:00:00")
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))

        for i in range(4):
            cls._add_post('p{}'.format(i), 'content', "2017-06-16 03:00:00")
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))

        cls._add_author('a2_longer', False, 0, created_at='Sat Dec 26 12:13:12 +0000 2009')
        for i in range(8, 12):
            cls._add_post('p{}'.format(i), 'content', "2017-07-18 05:00:00")
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))

        cls._add_author('a3_sh', created_at='Tue Dec 28 12:13:12 +0000 2009')
        for i in range(12, 16):
            cls._add_post('p{}'.format(i), 'content', "2017-08-18 04:00:00")
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))

        cls._add_claim('c2', "2017-06-15 03:00:00")
        cls._add_author('a4', followers=0)
        for i in range(40, 44):
            cls._add_post('p{}'.format(i), 'content', "2017-06-16 03:00:00")
            cls._add_claim_tweet_connection('c2', 'p{}'.format(i))

        cls._db.session.commit()

        cls._aggregated_authors_posts_feature_generator = AggregatedAuthorsPostsFeatureGenerator(cls._db,
                                                                                                 **{'authors': [],
                                                                                                    'posts': {}})
        cls._aggregated_authors_posts_feature_generator.execute()

    def tearDown(self):
        self._db.session.close()
        pass

    # def test_stress(self):
    #     claim_count = 5
    #     for j in xrange(1, claim_count):
    #         print 'add {}/{} claims'.format(j, claim_count)
    #         claim_index = 10 * j
    #         claim_id = u'c{}'.format(claim_index)
    #         self._add_claim(claim_id, "2017-06-15 03:00:00")
    #         self._add_author(u'a{}'.format(claim_index), created_at=u'Sat Dec 19 12:13:12 +0000 2009')
    #         for i in xrange(400 * j, 400 * (j + 1)):
    #             self._add_post(u'p{}'.format(i),
    #                            u'Seismic imaging processes are, in general, formulated under the assumption of a correct',
    #                            "2017-06-18 05:00:00")
    #             self._add_claim_tweet_connection(claim_id, u'p{}'.format(i))
    #     self._db.session.commit()
    #     self._aggregated_authors_posts_feature_generator = AggregatedAuthorsPostsFeatureGenerator(self._db,
    #                                                                                               **{'authors': [],
    #                                                                                                  'posts': {}})
    #     self._aggregated_authors_posts_feature_generator.execute()

    def test_authors_age_diff(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_authors_age_diff'
        self._generic_test('c1', '%s_sum' % base_feature_name, 9)
        self._generic_test('c1', '%s_mean' % base_feature_name, 9)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 9)
        self._generic_test('c1', '%s_max' % base_feature_name, 9)

        self._generic_test('c2', '%s_sum' % base_feature_name, 0)
        self._generic_test('c2', '%s_mean' % base_feature_name, 0)
        self._generic_test('c2', '%s_std' % base_feature_name, 0)
        self._generic_test('c2', '%s_min' % base_feature_name, 0)
        self._generic_test('c2', '%s_max' % base_feature_name, 0)

    def test_posts_date_diff(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_posts_date_diff'
        self._generic_test('c1', '%s_sum' % base_feature_name, 63)
        self._generic_test('c1', '%s_mean' % base_feature_name, 63)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 63)
        self._generic_test('c1', '%s_max' % base_feature_name, 63)

        self._generic_test('c2', '%s_sum' % base_feature_name, 0)
        self._generic_test('c2', '%s_mean' % base_feature_name, 0)
        self._generic_test('c2', '%s_std' % base_feature_name, 0)
        self._generic_test('c2', '%s_min' % base_feature_name, 0)
        self._generic_test('c2', '%s_max' % base_feature_name, 0)

    def test_author_screen_name_length(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_author_screen_name_length'
        self._generic_test('c1', '%s_sum' % base_feature_name, 20)
        self._generic_test('c1', '%s_mean' % base_feature_name, 6.666666667)
        self._generic_test('c1', '%s_std' % base_feature_name, 1.69967317)
        self._generic_test('c1', '%s_min' % base_feature_name, 5)
        self._generic_test('c1', '%s_max' % base_feature_name, 9)

        self._generic_test('c1', '%s_sum_0.8_newest' % base_feature_name, 14)
        self._generic_test('c1', '%s_mean_0.8_newest' % base_feature_name, 7)
        self._generic_test('c1', '%s_std_0.8_newest' % base_feature_name, 2)
        self._generic_test('c1', '%s_min_0.8_newest' % base_feature_name, 5)
        self._generic_test('c1', '%s_max_0.8_newest' % base_feature_name, 9)

    def test_posts_age(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_posts_age'
        self._generic_test('c1', '%s_sum' % base_feature_name, 404)
        self._generic_test('c1', '%s_mean' % base_feature_name, 25.25)
        self._generic_test('c1', '%s_std' % base_feature_name, 25.7135664581948)
        self._generic_test('c1', '%s_min' % base_feature_name, 1)
        self._generic_test('c1', '%s_max' % base_feature_name, 64)

        self._generic_test('c2', '%s_sum' % base_feature_name, 4)
        self._generic_test('c2', '%s_mean' % base_feature_name, 1)
        self._generic_test('c2', '%s_std' % base_feature_name, 0)
        self._generic_test('c2', '%s_min' % base_feature_name, 1)
        self._generic_test('c2', '%s_max' % base_feature_name, 1)

    def test_num_of_retweets(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_retweets'
        self._generic_test('c2', '%s_sum' % base_feature_name, 16)
        self._generic_test('c2', '%s_mean' % base_feature_name, 4)
        self._generic_test('c2', '%s_std' % base_feature_name, 0)
        self._generic_test('c2', '%s_min' % base_feature_name, 4)
        self._generic_test('c2', '%s_max' % base_feature_name, 4)

    def test_posts_num_of_favorites(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_posts_num_of_favorites'
        self._generic_test('c2', '%s_sum' % base_feature_name, 12)
        self._generic_test('c2', '%s_mean' % base_feature_name, 3)
        self._generic_test('c2', '%s_std' % base_feature_name, 0)
        self._generic_test('c2', '%s_min' % base_feature_name, 3)
        self._generic_test('c2', '%s_max' % base_feature_name, 3)

    def test_num_of_followers(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_followers'
        self._generic_test('c1', '%s_sum' % base_feature_name, 30)
        self._generic_test('c1', '%s_mean' % base_feature_name, 10)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 10)
        self._generic_test('c1', '%s_max' % base_feature_name, 10)

    def test_num_of_friends(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_friends'
        self._generic_test('c1', '%s_sum' % base_feature_name, 15)
        self._generic_test('c1', '%s_mean' % base_feature_name, 5)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 5)
        self._generic_test('c1', '%s_max' % base_feature_name, 5)

    def test_num_of_statuses(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_statuses'
        self._generic_test('c1', '%s_sum' % base_feature_name, 18)
        self._generic_test('c1', '%s_mean' % base_feature_name, 6)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 6)
        self._generic_test('c1', '%s_max' % base_feature_name, 6)

    def test_num_of_favorites(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_statuses'
        self._generic_test('c1', '%s_sum' % base_feature_name, 18)
        self._generic_test('c1', '%s_mean' % base_feature_name, 6)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 6)
        self._generic_test('c1', '%s_max' % base_feature_name, 6)

    def test_num_of_listed_count(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_statuses'
        self._generic_test('c1', '%s_sum' % base_feature_name, 18)
        self._generic_test('c1', '%s_mean' % base_feature_name, 6)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 6)
        self._generic_test('c1', '%s_max' % base_feature_name, 6)

    def test_num_of_protected(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_protected'
        self._generic_test('c1', '%s_sum' % base_feature_name, 2)
        self._generic_test('c1', '%s_mean' % base_feature_name, 2.0 / 3)
        self._generic_test('c1', '%s_std' % base_feature_name, 0.4714045207)
        self._generic_test('c1', '%s_min' % base_feature_name, 0)
        self._generic_test('c1', '%s_max' % base_feature_name, 1)

    def test_num_of_verified(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_num_of_verified'
        self._generic_test('c1', '%s_sum' % base_feature_name, 2)
        self._generic_test('c1', '%s_mean' % base_feature_name, 2.0 / 3)
        self._generic_test('c1', '%s_std' % base_feature_name, 0.4714045207)
        self._generic_test('c1', '%s_min' % base_feature_name, 0)
        self._generic_test('c1', '%s_max' % base_feature_name, 1)

    def test_friends_followers_ratio(self):
        base_feature_name = 'AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio'
        self._generic_test('c1', '%s_sum' % base_feature_name, 1.5)
        self._generic_test('c1', '%s_mean' % base_feature_name, 0.5)
        self._generic_test('c1', '%s_std' % base_feature_name, 0)
        self._generic_test('c1', '%s_min' % base_feature_name, 0.5)
        self._generic_test('c1', '%s_max' % base_feature_name, 0.5)

        self._generic_test('c2', '%s_sum' % base_feature_name, -1)
        self._generic_test('c2', '%s_mean' % base_feature_name, -1)
        self._generic_test('c2', '%s_std' % base_feature_name, -1)
        self._generic_test('c2', '%s_min' % base_feature_name, -1)
        self._generic_test('c2', '%s_max' % base_feature_name, -1)

    def _generic_test(self, author_guid, attribute, expected_value):
        db_val = self._db.get_author_feature(author_guid, attribute).attribute_value
        self.assertAlmostEqual(eval(db_val), float(expected_value))

    @classmethod
    def _add_author(cls, author_guid, protected=True, verified=1, created_at='Mon Oct 12 12:40:21 +0000 2015',
                    followers=10):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = author_guid
        author.author_screen_name = author_guid
        author.name = author_guid
        author.created_at = created_at
        author.domain = 'tests'
        author.statuses_count = 6
        author.followers_count = followers
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
        cls._db.addPost(post)
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
