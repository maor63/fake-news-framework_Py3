from collections import defaultdict

import pandas as pd
import sys

from dataset_builder.feature_extractor.reddit_feature_generator import RedditByClaimFeatureGenerator

__author__ = "Joshua Grogin"


class RedditAuthorByClaimFeatureGenerator(RedditByClaimFeatureGenerator):
    def __init__(self, db, **kwargs):
        RedditByClaimFeatureGenerator.__init__(self, db, **kwargs)

    def link_karma(self, claim_id, reddit_authors):
        return self.apply_aggregation_functions(reddit_authors, claim_id, 'link_karma', self._func_name())

    def comment_karma(self, claim_id, reddit_authors):
        return self.apply_aggregation_functions(reddit_authors, claim_id, 'comment_karma', self._func_name())

    def total_karma(self, claim_id, reddit_authors):
        reddit_authors = [type('', (object,), {"karma": reddit_author.link_karma + reddit_author.comment_karma})() for
                          reddit_author in reddit_authors]
        return self.apply_aggregation_functions(reddit_authors, claim_id, 'karma', self._func_name())

    def count_is_gold(self, claim_id, reddit_authors):
        return self.apply_count_aggregation_functions(reddit_authors, claim_id, 'is_gold', self._func_name())

    def count_is_moderator(self, claim_id, reddit_authors):
        return self.apply_count_aggregation_functions(reddit_authors, claim_id, 'is_moderator', self._func_name())

    def count_is_employee(self, claim_id, reddit_authors):
        return self.apply_count_aggregation_functions(reddit_authors, claim_id, 'is_employee', self._func_name())

    def ratio_is_gold(self, claim_id, reddit_authors):
        return self.apply_ratio_aggregation_functions(reddit_authors, claim_id, 'is_gold', self._func_name())

    def ratio_is_moderator(self, claim_id, reddit_authors):
        return self.apply_ratio_aggregation_functions(reddit_authors, claim_id, 'is_moderator', self._func_name())

    def ratio_is_employee(self, claim_id, reddit_authors):
        return self.apply_ratio_aggregation_functions(reddit_authors, claim_id, 'is_employee', self._func_name())

    def apply_count_aggregation_functions(self, target_elements, source_id, field, name):
        field_series = [getattr(target_element, field, None) for target_element in target_elements
                        if getattr(target_element, field, None) is not None]
        pd_series = pd.Series(field_series)
        return [self._create_author_feature('', name, source_id, pd_series.sum())]

    def apply_ratio_aggregation_functions(self, target_elements, source_id, field, name):
        field_series = [getattr(target_element, field, None) for target_element in target_elements
                        if getattr(target_element, field, None) is not None]
        pd_series = pd.Series(field_series)
        return [self._create_author_feature('', name, source_id, self._ratio(pd_series))]

    def _get_reddit_authors_claims_connections(self):
        claims_post_author_connection = self._db.get_claim_post_author_connections()
        reddit_authors = dict((author.author_guid, author) for author in self._db.get_reddit_authors())
        source_id_target_elements_dict = defaultdict(set)
        for claim_id, post_id, author_guid in claims_post_author_connection:
            if author_guid in reddit_authors:
                source_id_target_elements_dict[claim_id].add(reddit_authors[author_guid])
        return source_id_target_elements_dict

    def _get_reddit_posts_claims_connections(self):
        return self._get_reddit_authors_claims_connections()

    def _ratio(self, pd_series):
        return pd_series.sum() / float(pd_series.count())
