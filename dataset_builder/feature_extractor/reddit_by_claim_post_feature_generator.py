from collections import defaultdict
from dataset_builder.feature_extractor.reddit_feature_generator import RedditByClaimFeatureGenerator

__author__ = "Joshua Grogin"


class RedditPostByClaimFeatureGenerator(RedditByClaimFeatureGenerator):
    def __init__(self, db, **kwargs):
        RedditByClaimFeatureGenerator.__init__(self, db, **kwargs)

    def karma_by_submission_and_comment(self, claim_id, reddit_posts):
        return self.apply_aggregation_functions(reddit_posts, claim_id, 'score', self._func_name())

    def karma_by_submission(self, claim_id, reddit_posts):
            return self.apply_aggregation_functions(self._remove_comments(reddit_posts), claim_id, 'score', self._func_name())

    def upvotes_by_submission(self, claim_id, reddit_posts):
            return self.apply_aggregation_functions(self._remove_comments(reddit_posts), claim_id, 'ups', self._func_name())

    def downvotes_by_submission(self, claim_id, reddit_posts):
            return self.apply_aggregation_functions(self._remove_comments(reddit_posts), claim_id, 'downs', self._func_name())

    def _remove_comments(self, reddit_posts):
        return [post for post in reddit_posts if post.ups != -1]

    def _get_reddit_posts_claims_connections(self):
        claims_post_connection = self._db.get_claim_post_author_connections()
        reddit_posts = dict((post.guid, post) for post in self._db.get_reddit_posts())
        source_id_target_elements_dict = defaultdict(set)
        for claim_id, post_id, author_guid in claims_post_connection:
            if post_id in reddit_posts:
                source_id_target_elements_dict[claim_id].add(reddit_posts[post_id])
        return source_id_target_elements_dict