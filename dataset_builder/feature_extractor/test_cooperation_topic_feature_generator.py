from unittest import TestCase
from DB.schema_definition import *
from dataset_builder.feature_extractor.cooperation_topic_feature_generator import CooperationTopicFeatureGenerator


class TestCooperationTopicFeatureGenerator(TestCase):
    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestCooperationTopicFeatureGenerator, cls).setUpClass()
        cls._db = DB()
        cls._db.setUp()
        cls._posts = []
        cls._post_dictionary = {}

        cls._add_claim('c1', "2017-06-15 03:00:00")
        cls._add_author('a1', created_at='Sat Dec 19 12:13:12 +0000 2009')
        for i in range(4):
            cls._add_post('p{}'.format(i),
                          'content similar content one nice word Permutation Combination Python hello good bad',
                          "2017-06-16 03:00:00")
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))

        cls._add_author('a2', created_at='Sat Dec 19 12:13:12 +0000 2009')
        for i in range(4, 8):
            cls._add_post('p{}'.format(i),
                          'content similar content one nice word Permutation Combination hello good bad',
                          "2017-06-18 05:00:00")
            cls._add_claim_tweet_connection('c1', 'p{}'.format(i))
        cls._add_author('a3', False, 0, created_at='Sat Dec 26 12:13:12 +0000 2009')
        for i in range(8, 12):
            cls._add_post('p{}'.format(i),
                          'content similar content one nice word Permutation Combination Python hello good bad',
                          "2017-07-18 05:00:00")
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
        cls._add_claim('c3', "2017-06-15 03:00:00")
        cls._add_author('a7', created_at='Tue Dec 28 12:13:12 +0000 2009')
        cls._add_post('p{}'.format(47), 'skldf sdlfkjsldfj sldkfjlsdfj sdfjiebvcki', "2017-06-16 03:00:00")
        cls._add_claim_tweet_connection('c3', 'p{}'.format(47))
        cls._add_author('a8', created_at='Tue Dec 28 12:13:12 +0000 2009')
        cls._db.session.commit()

        cls._cooperation_feature_generator = CooperationTopicFeatureGenerator(cls._db, **{'authors': [], 'posts': {}})
        cls._cooperation_feature_generator.setUp()
        #

    def tearDown(self):
        self._db.session.close()
        pass

    def test_execute(self):
        self._cooperation_feature_generator.execute()
        base_feature_name = 'CooperationTopicFeatureGenerator_authors_cooperation'
        self._generic_test('c1', '%s_sum_jaccard_0.9' % base_feature_name, 6)
        self._generic_test('c1', '%s_mean_jaccard_0.9' % base_feature_name, 2)
        self._generic_test('c1', '%s_median_jaccard_0.9' % base_feature_name, 2)
        self._generic_test('c1', '%s_std_jaccard_0.9' % base_feature_name, 0)
        self._generic_test('c1', '%s_min_jaccard_0.9' % base_feature_name, 2)
        self._generic_test('c1', '%s_max_jaccard_0.9' % base_feature_name, 2)

    def test_calculate_topics_similarity(self):
        post_id_to_words = self._cooperation_feature_generator._create_post_id_to_content_words(self._posts)
        authors_counter_dic = self._cooperation_feature_generator.calculate_author_cooperation(post_id_to_words)
        self.assertSetEqual(authors_counter_dic['a1'], {'a2', 'a3'})
        self.assertSetEqual(authors_counter_dic['a2'], {'a1', 'a3'})
        self.assertSetEqual(authors_counter_dic['a3'], {'a2', 'a1'})
        self.assertSetEqual(authors_counter_dic['a4'], {'a5'})
        self.assertSetEqual(authors_counter_dic['a5'], {'a4'})
        self.assertSetEqual(authors_counter_dic['a6'], set())

    def test_calculate_topics_exact_match(self):
        post_id_to_words = self._cooperation_feature_generator._create_post_id_to_content_words(self._posts)
        authors_counter_dic = self._cooperation_feature_generator.calculate_author_cooperation(post_id_to_words, 1.0)
        self.assertSetEqual(authors_counter_dic['a1'], {'a3'})
        self.assertSetEqual(authors_counter_dic['a2'], set())
        self.assertSetEqual(authors_counter_dic['a3'], {'a1'})
        self.assertSetEqual(authors_counter_dic['a4'], {'a5'})
        self.assertSetEqual(authors_counter_dic['a5'], {'a4'})
        self.assertSetEqual(authors_counter_dic['a6'], set())

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
