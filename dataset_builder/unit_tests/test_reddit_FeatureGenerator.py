from unittest import TestCase

from DB.schema_definition import DB, Post, Author, Claim_Tweet_Connection, RedditAuthor, RedditPost, Claim
from commons.commons import convert_str_to_unicode_datetime, compute_author_guid_by_author_name, compute_post_guid, \
    date_to_str, str_to_date
from dataset_builder.feature_extractor.reddit_by_claim_post_feature_generator import RedditPostByClaimFeatureGenerator
from dataset_builder.feature_extractor.reddit_by_claim_author_feature_generator import \
    RedditAuthorByClaimFeatureGenerator


class RedditFeatureGeneratorTest(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._posts = []
        self._author = None
        self._init_authors()
        self._init_posts()
        self._init_claims()
        self._reddit_post_by_claim_feature_generator = RedditPostByClaimFeatureGenerator(self._db, **self._get_params())
        self._reddit_author_by_claim_feature_generator = RedditAuthorByClaimFeatureGenerator(self._db, **self._get_params())

    def tearDown(self):
        self._db.session.close()
        pass

    def test_karma_by_submission_and_comment(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'min_karma_by_submission_and_comment',
                'expected': -13
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'min_karma_by_submission_and_comment',
                'expected': -321
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'min_karma_by_submission_and_comment',
                'expected': 1
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'max_karma_by_submission_and_comment',
                'expected': 52312
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'max_karma_by_submission_and_comment',
                'expected': 102
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'max_karma_by_submission_and_comment',
                'expected': 234
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'mean_karma_by_submission_and_comment',
                'expected': 5904.222222
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'mean_karma_by_submission_and_comment',
                'expected': -19.55555556
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'mean_karma_by_submission_and_comment',
                'expected': 38.5
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'median_karma_by_submission_and_comment',
                'expected': 27
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'median_karma_by_submission_and_comment',
                'expected': 7
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'median_karma_by_submission_and_comment',
                'expected': 5
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'skew_karma_by_submission_and_comment',
                'expected': 2.998904337
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'skew_karma_by_submission_and_comment',
                'expected': -2.525365088
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'skew_karma_by_submission_and_comment',
                'expected': 2.234762661
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'kurtosis_karma_by_submission_and_comment',
                'expected': 8.995080203
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'kurtosis_karma_by_submission_and_comment',
                'expected': 7.357797068
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'kurtosis_karma_by_submission_and_comment',
                'expected': 4.503581242
            }
        ]
        self._reddit_post_by_claim_feature_generator._measure_names = ['karma_by_submission_and_comment']
        self._reddit_post_by_claim_feature_generator._aggregation_functions = ['min', 'max', 'mean', 'median', 'skew',
                                                                               'kurtosis']
        self._reddit_post_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_post_by_claim_feature_generator.__class__.__name__)

    def test_karma_by_submission(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'min_karma_by_submission',
                'expected': 738
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'min_karma_by_submission',
                'expected': -321
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'min_karma_by_submission',
                'expected': 123
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'max_karma_by_submission',
                'expected': 52312
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'max_karma_by_submission',
                'expected': 102
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'max_karma_by_submission',
                'expected': 234
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'mean_karma_by_submission',
                'expected': 26525
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'mean_karma_by_submission',
                'expected': -109.5
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'mean_karma_by_submission',
                'expected': 178.5
            },
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': 'median_karma_by_submission',
                'expected': 26525
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': 'median_karma_by_submission',
                'expected': -109.5
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': 'median_karma_by_submission',
                'expected': 178.5
            }
        ]
        self._reddit_post_by_claim_feature_generator._measure_names = ['karma_by_submission']
        self._reddit_post_by_claim_feature_generator._aggregation_functions = ['min', 'max', 'mean', 'median', 'skew',
                                                                               'kurtosis']
        self._reddit_post_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_post_by_claim_feature_generator.__class__.__name__)

    def test_upvotes_by_submission(self):
        test_cases = [
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'min_upvotes_by_submission',
                        'expected': 762
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'min_upvotes_by_submission',
                        'expected': 112
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'min_upvotes_by_submission',
                        'expected': 369
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'max_upvotes_by_submission',
                        'expected': 74593
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'max_upvotes_by_submission',
                        'expected': 241
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'max_upvotes_by_submission',
                        'expected': 2067
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'mean_upvotes_by_submission',
                        'expected': 37677.5
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'mean_upvotes_by_submission',
                        'expected': 176.5
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'mean_upvotes_by_submission',
                        'expected': 1218
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'median_upvotes_by_submission',
                        'expected': 37677.5
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'median_upvotes_by_submission',
                        'expected': 176.5
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'median_upvotes_by_submission',
                        'expected': 1218
                    }
                ]
        self._reddit_post_by_claim_feature_generator._measure_names = ['upvotes_by_submission']
        self._reddit_post_by_claim_feature_generator._aggregation_functions = ['min', 'max', 'mean', 'median', 'skew',
                                                                               'kurtosis']
        self._reddit_post_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_post_by_claim_feature_generator.__class__.__name__)

    def test_downvotes_by_submission(self):
        test_cases = [
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'min_downvotes_by_submission',
                        'expected': 24
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'min_downvotes_by_submission',
                        'expected': 10
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'min_downvotes_by_submission',
                        'expected': 246
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'max_downvotes_by_submission',
                        'expected': 22281
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'max_downvotes_by_submission',
                        'expected': 562
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'max_downvotes_by_submission',
                        'expected': 1833
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'mean_downvotes_by_submission',
                        'expected': 11152.5
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'mean_downvotes_by_submission',
                        'expected': 286
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'mean_downvotes_by_submission',
                        'expected': 1039.5
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'median_downvotes_by_submission',
                        'expected': 11152.5
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'median_downvotes_by_submission',
                        'expected': 286
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'median_downvotes_by_submission',
                        'expected': 1039.5
                    }
                ]
        self._reddit_post_by_claim_feature_generator._measure_names = ['downvotes_by_submission']
        self._reddit_post_by_claim_feature_generator._aggregation_functions = ['min', 'max', 'mean', 'median', 'skew',
                                                                               'kurtosis']
        self._reddit_post_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_post_by_claim_feature_generator.__class__.__name__)

    def test_author_comment_karma(self):
        test_cases = [
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'min_comment_karma',
                        'expected': 2261
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'min_comment_karma',
                        'expected': 2842
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'min_comment_karma',
                        'expected': 2842
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'max_comment_karma',
                        'expected': 37027
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'max_comment_karma',
                        'expected': 35111
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'max_comment_karma',
                        'expected': 30880
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'mean_comment_karma',
                        'expected': 19096.66667
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'mean_comment_karma',
                        'expected': 18031
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'mean_comment_karma',
                        'expected': 11833.5
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'median_comment_karma',
                        'expected': 22588
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'median_comment_karma',
                        'expected': 16555
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'median_comment_karma',
                        'expected': 6806
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'skew_comment_karma',
                        'expected': -0.018614054
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'skew_comment_karma',
                        'expected': 0.128211429
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'skew_comment_karma',
                        'expected': 1.862860226
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'kurtosis_comment_karma',
                        'expected': -1.992620739
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'kurtosis_comment_karma',
                        'expected': -2.723581645
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'kurtosis_comment_karma',
                        'expected': 3.595027437
                    }
                ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['comment_karma']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = ['min', 'max', 'mean', 'median', 'skew', 'kurtosis']
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_link_karma(self):
        test_cases = [
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'min_link_karma',
                        'expected': 1
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'min_link_karma',
                        'expected': 1
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'min_link_karma',
                        'expected': 90
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'max_link_karma',
                        'expected': 171576
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'max_link_karma',
                        'expected': 171576
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'max_link_karma',
                        'expected': 5897
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'mean_link_karma',
                        'expected': 20565.77778
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'mean_link_karma',
                        'expected': 29840.16667
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'mean_link_karma',
                        'expected': 1866
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'median_link_karma',
                        'expected': 1341
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'median_link_karma',
                        'expected': 738.5
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'median_link_karma',
                        'expected': 738.5
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'skew_link_karma',
                        'expected': 2.991811692
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'skew_link_karma',
                        'expected': 2.443747273
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'skew_link_karma',
                        'expected': 1.751305522
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'kurtosis_link_karma',
                        'expected': 8.963145712
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'kurtosis_link_karma',
                        'expected': 5.977609271
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'kurtosis_link_karma',
                        'expected': 3.018013716
                    }
                ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['link_karma']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = ['min', 'max', 'mean', 'median', 'skew', 'kurtosis']
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_total_karma(self):
        test_cases = [
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'min_total_karma',
                        'expected': 2435
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'min_total_karma',
                        'expected': 6379
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'min_total_karma',
                        'expected': 6379
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'max_total_karma',
                        'expected': 206687
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'max_total_karma',
                        'expected':  206687
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'max_total_karma',
                        'expected': 32221
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'mean_total_karma',
                        'expected': 39662.44444
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'mean_total_karma',
                        'expected': 47871.16667
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'mean_total_karma',
                        'expected': 13699.5
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'median_total_karma',
                        'expected': 22589
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'median_total_karma',
                        'expected': 17240.5
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'median_total_karma',
                        'expected': 8099
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'skew_total_karma',
                        'expected': 2.767953592
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'skew_total_karma',
                        'expected': 2.349097328
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'skew_total_karma',
                        'expected': 1.963784833
                    },
                    {
                        'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                        'test_name': 'kurtosis_total_karma',
                        'expected': 7.954685555
                    },
                    {
                        'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                        'test_name': 'kurtosis_total_karma',
                        'expected': 5.605190323
                    },
                    {
                        'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                        'test_name': 'kurtosis_total_karma',
                        'expected': 3.878351431
                    }
                ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['total_karma']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = ['min', 'max', 'mean', 'median', 'skew', 'kurtosis']
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_count_is_gold(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': '_count_is_gold',
                'expected': 3
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': '_count_is_gold',
                'expected': 3
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': '_count_is_gold',
                'expected': 3
            }
        ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['count_is_gold']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = []
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_count_is_moderator(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': '_count_is_moderator',
                'expected': 2
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': '_count_is_moderator',
                'expected': 1
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': '_count_is_moderator',
                'expected': 0
            }
        ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['count_is_moderator']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = []
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_count_is_employee(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': '_count_is_employee',
                'expected': 3
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': '_count_is_employee',
                'expected': 1
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': '_count_is_employee',
                'expected': 1
            }
        ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['count_is_employee']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = []
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_ratio_is_gold(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': '_ratio_is_gold',
                'expected': 0.333333333
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': '_ratio_is_gold',
                'expected': 0.5
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': '_ratio_is_gold',
                'expected': 0.75
            }
        ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['ratio_is_gold']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = []
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_ratio_is_moderator(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': '_ratio_is_moderator',
                'expected': 0.222222222
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': '_ratio_is_moderator',
                'expected': 0.166666667
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': '_ratio_is_moderator',
                'expected': 0
            }
        ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['ratio_is_moderator']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = []
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def test_author_ratio_is_employee(self):
        test_cases = [
            {
                'claim_id': 'cd2e1978-4dfa-3a40-b62f-71153001629c',
                'test_name': '_ratio_is_employee',
                'expected': 0.333333333
            },
            {
                'claim_id': 'a4beae51-463f-33fc-bbf6-20eca5104afe',
                'test_name': '_ratio_is_employee',
                'expected': 0.166666667
            },
            {
                'claim_id': '9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                'test_name': '_ratio_is_employee',
                'expected': 0.25
            }
        ]
        self._reddit_author_by_claim_feature_generator._measure_names = ['ratio_is_employee']
        self._reddit_author_by_claim_feature_generator._aggregation_functions = []
        self._reddit_author_by_claim_feature_generator.execute()

        for test_case in test_cases:
            self.assert_author_feature_test_case(test_case, self._reddit_author_by_claim_feature_generator.__class__.__name__)

    def assert_author_feature_test_case(self, test_case, class_name):
        self.assert_author_feature_number(test_case['claim_id'],
                                       "{}_{}".format(
                                           class_name,
                                           test_case['test_name']), test_case['expected'])

    def assert_author_feature_number(self, author_guid, attribute_name, expected):
        result_feature = self._db.get_author_feature(author_guid, attribute_name)
        feature_value = getattr(result_feature, 'attribute_value')
        self.assertAlmostEqual(float(expected), float(feature_value), places=2)

    def _add_author(self, name=None, link_karma=None, comment_karma=None, is_employee=0, is_mod=0, is_gold=0,
                    author_osn_id=None):
        author = Author()
        reddit_author = RedditAuthor()
        author.name = name
        author.author_screen_name = author.name
        author.author_guid = compute_author_guid_by_author_name(author.name)
        author.domain = 'reddit'
        author.author_osn_id = author_osn_id
        author.author_full_name = name
        author.url = 'https://www.reddit.com/user/' + name

        reddit_author.name = author.name
        reddit_author.author_guid = author.author_guid

        reddit_author.comments_count = None
        reddit_author.comment_karma = comment_karma
        reddit_author.link_karma = link_karma
        reddit_author.is_gold = is_gold
        reddit_author.is_moderator = is_mod
        reddit_author.is_employee = is_employee

        self._db.add_authors([author])
        self._db.add_reddit_authors([reddit_author])
        # self._author = author

    def _add_post(self, author, date, post_osn_id, score=0, upvote_ratio=-1):
        post = Post()
        post.post_osn_id = post_osn_id
        post.author = str(author)
        post.author_guid = compute_author_guid_by_author_name(post.author)
        post.created_at = str_to_date(date, formate="%d/%m/%Y %H:%M")
        post.url = 'https://www.reddit.com{}'.format(post.author)  # just for test
        post.guid = compute_post_guid(post.url, post.post_osn_id, date_to_str(post.created_at))
        post.domain = 'reddit_comment'
        post.post_type = 'reddit_comment'
        post.post_id = post.guid

        reddit_post = RedditPost()
        reddit_post.post_id = post.post_id
        reddit_post.guid = post.guid
        reddit_post.score = score
        if upvote_ratio != -1:
            post.domain = 'reddit_post'
            post.post_type = 'reddit_post'
            reddit_post.upvote_ratio = upvote_ratio
            reddit_post.ups = int(
                round((reddit_post.upvote_ratio * reddit_post.score) / (2 * reddit_post.upvote_ratio - 1))
                if reddit_post.upvote_ratio != 0.5
                else round(reddit_post.score / 2))
            reddit_post.downs = reddit_post.ups - reddit_post.score
        else:
            reddit_post.ups = -1
            reddit_post.downs = -1
            reddit_post.upvote_ratio = -1

        self._db.addPosts([post, reddit_post])
        return post, reddit_post

    def _add_claim_tweet_connection(self, claim_id, post_id):
        connection = Claim_Tweet_Connection()
        connection.claim_id = claim_id
        connection.post_id = post_id
        self._db.add_claim_connections([connection])

    def _add_claim(self, claim_id):
        claim = Claim()
        claim.claim_id = claim_id
        self._db.addPosts([claim])

    def _init_authors(self):
        self._add_author('Smile_lifeisgood', comment_karma=30880, link_karma=1341, is_gold=1, is_mod=0, is_employee=0)
        self._add_author('Cunty_Balls', comment_karma=7369, link_karma=90, is_gold=1, is_mod=0, is_employee=0)
        self._add_author('I_kick_fuck_nuns', comment_karma=2842, link_karma=5897, is_gold=1, is_mod=0, is_employee=0)
        self._add_author('TheRiseofMindhawk', comment_karma=2261, link_karma=174, is_gold=1, is_mod=1, is_employee=0)
        self._add_author('dialog2011', comment_karma=37027, link_karma=4582, is_gold=0, is_mod=0, is_employee=1)
        self._add_author('chrmanyaki', comment_karma=22588, link_karma=1, is_gold=0, is_mod=0, is_employee=1)
        self._add_author('Undertakerjoe', comment_karma=9177, link_karma=1384, is_gold=0, is_mod=0, is_employee=0)
        self._add_author('Lmb2298', comment_karma=25741, link_karma=1, is_gold=0, is_mod=0, is_employee=0)
        self._add_author('azzazaz', comment_karma=35111, link_karma=171576, is_gold=0, is_mod=1, is_employee=0)
        self._add_author('juanwonone1', comment_karma=6243, link_karma=136, is_gold=0, is_mod=0, is_employee=1)

    def _init_posts(self):
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('juanwonone1', '15/10/2017 21:44', '76ksr4', 738, 0.97)[
                                             0].guid)
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('Lmb2298', '01/10/2017 22:24', 'dferfgh', 52312, 0.77)[
                                             0].guid)

        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('dialog2011', '12/06/2017 23:45', '6gv0vk', 27)[0].guid)
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('chrmanyaki', '15/10/2017 21:58', 'doeq8ke', 27)[0].guid)
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('azzazaz', '12/06/2018 10:50', 'e0j4zkz', 32)[0].guid)
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('Smile_lifeisgood', '12/06/2018 20:08', 'e0in2zm', 11)[
                                             0].guid)
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('Undertakerjoe', '15/10/2017 22:17', 'doerbqu', -13)[0].guid)
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('I_kick_fuck_nuns', '18/06/2017 3:39', 'dj1qid5', 2)[0].guid)
        self._add_claim_tweet_connection('cd2e1978-4dfa-3a40-b62f-71153001629c',
                                         self._add_post('TheRiseofMindhawk', '13/06/2017 8:17', 'ditymrc', 2)[
                                             0].guid)

        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('I_kick_fuck_nuns', '11/06/2018 18:49', '8qal3m', 102, 0.92)[
                                             0].guid)
        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('juanwonone1', '16/10/2017 2:23', 'dof4fen', -321, 0.3)[
                                             0].guid)

        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('Smile_lifeisgood', '13/06/2017 0:29', 'dditbt8r', 11)[
                                             0].guid)
        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('Lmb2298', '15/10/2017 22:38', 'doeslie', 11)[0].guid)
        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('azzazaz', '16/10/2017 0:30', 'doeyvtb', 9)[0].guid)
        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('juanwonone1', '15/10/2017 22:50', 'doetc6j', 7)[0].guid)
        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('Cunty_Balls', '16/10/2017 1:52', 'dof2x1x', 2)[0].guid)
        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('Cunty_Balls', '16/10/2017 2:43', 'dof5cpo', 2)[0].guid)
        self._add_claim_tweet_connection('a4beae51-463f-33fc-bbf6-20eca5104afe',
                                         self._add_post('juanwonone1', '16/10/2017 3:45', 'dof84f8', 1)[0].guid)

        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('Cunty_Balls', '15/10/2017 22:24', 'doerqsj', 234, 0.53)[
                                             0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('I_kick_fuck_nuns', '16/10/2017 21:44', '76ksr2', 123, 0.6)[
                                             0].guid)

        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('Smile_lifeisgood', '13/06/2017 7:04', 'ditvpox', 7)[0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('Smile_lifeisgood', '13/06/2017 0:51', 'ditcy28', 5)[0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('juanwonone1', '15/10/2017 23:36', 'doevzsq', 5)[0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('juanwonone1', '16/10/2017 0:26', 'doeynrr', 5)[0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('I_kick_fuck_nuns', '11/06/2018 21:55', 'e0hy5he', 1)[
                                             0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('I_kick_fuck_nuns', '11/06/2018 22:04', 'e0hyrhi', 1)[
                                             0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('I_kick_fuck_nuns', '12/06/2018 1:31', 'e0icveq', 1)[0].guid)
        self._add_claim_tweet_connection('9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
                                         self._add_post('Cunty_Balls', '13/06/2017 7:55', 'ditxua6', 3)[0].guid)

    def _init_claims(self):
        self._add_claim('cd2e1978-4dfa-3a40-b62f-71153001629c')
        self._add_claim('a4beae51-463f-33fc-bbf6-20eca5104afe')
        self._add_claim('9e875999-9a3e-3357-bfa6-ede4fe67c1c9')

    def _get_params(self):
        return {'authors': [], 'posts': []}


# test_cases = [
#             {
#                 'claim_id': u'cd2e1978-4dfa-3a40-b62f-71153001629c',
#                 'test_name': 'min_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'a4beae51-463f-33fc-bbf6-20eca5104afe',
#                 'test_name': 'min_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
#                 'test_name': 'min_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'cd2e1978-4dfa-3a40-b62f-71153001629c',
#                 'test_name': 'max_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'a4beae51-463f-33fc-bbf6-20eca5104afe',
#                 'test_name': 'max_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
#                 'test_name': 'max_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'cd2e1978-4dfa-3a40-b62f-71153001629c',
#                 'test_name': 'mean_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'a4beae51-463f-33fc-bbf6-20eca5104afe',
#                 'test_name': 'mean_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
#                 'test_name': 'mean_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'cd2e1978-4dfa-3a40-b62f-71153001629c',
#                 'test_name': 'median_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'a4beae51-463f-33fc-bbf6-20eca5104afe',
#                 'test_name': 'median_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
#                 'test_name': 'median_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'cd2e1978-4dfa-3a40-b62f-71153001629c',
#                 'test_name': 'skew_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'a4beae51-463f-33fc-bbf6-20eca5104afe',
#                 'test_name': 'skew_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
#                 'test_name': 'skew_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'cd2e1978-4dfa-3a40-b62f-71153001629c',
#                 'test_name': 'kurtosis_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'a4beae51-463f-33fc-bbf6-20eca5104afe',
#                 'test_name': 'kurtosis_',
#                 'expected':
#             },
#             {
#                 'claim_id': u'9e875999-9a3e-3357-bfa6-ede4fe67c1c9',
#                 'test_name': 'kurtosis_',
#                 'expected':
#             }
#         ]