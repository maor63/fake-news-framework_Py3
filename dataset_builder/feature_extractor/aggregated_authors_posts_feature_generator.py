
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from dateutil import parser
from DB.schema_definition import Claim
from commons.commons import *
from itertools import product
import datetime
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator


def contains_posts_decorator(func):
    def func_wrapper(*args, **kwargs):
        if 'posts' not in kwargs:
            raise ValueError('{} feature require posts as destination'.format(func))
        return func(*args, **kwargs)

    return func_wrapper


def fraction_results_decorator(func):
    def func_wrapper(self, *args, **kwargs):
        value_list = func(self, *args, **kwargs)
        fraction = max(int(self._division_percent * len(value_list)), 1)
        return value_list[:fraction]

    return func_wrapper


class AggregatedAuthorsPostsFeatureGenerator(BaseFeatureGenerator):

    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._aggregated_functions = self._config_parser.eval(self.__class__.__name__, "aggregated_functions")
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._division_percents = self._config_parser.eval(self.__class__.__name__, "division_percents")
        self._features = self._config_parser.eval(self.__class__.__name__, "feature_list")
        self._curr_aggregated_function = None
        self._division_percent = None
        self.authors_df = None
        self._clear_memory()
        self._features_names_count = len(self._features) * len(self._aggregated_functions) * len(
            self._division_percents)

    def _clear_memory(self):
        self._source_authors_dict = defaultdict()

    # def setUp(self):
    #     self._db.delete_class_author_features(self.__class__.__name__)

    def execute(self, window_start=None):
        # self._author_dict = self._db.get_author_dictionary()
        authors_dfs = []
        for authors_df in pd.read_sql_table('authors', self._db.engine, chunksize=10000):
            print('\r load authors {}'.format(len(authors_df)), end='')
            authors_dfs.append(authors_df)
        print()
        self.authors_df = pd.concat(authors_dfs).set_index('author_guid')
        # del authors_dfs

        self._division_percents = set(self._division_percents)
        self._division_percents.add(1)
        aggregated_functions = list(map(eval, self._aggregated_functions))
        total_authors_features = []
        seen_author_features = set()
        computed_features_names = []

        for targeted_fields_dict in self._targeted_fields:
            for source_id_source_element_dict, source_id_target_elements_dict in self._load_data_using_arg(
                    targeted_fields_dict):
                self._clear_memory()
                print()
                for i, self._division_percent in enumerate(self._division_percents):
                    if 0 < self._division_percent <= 1:
                        suffix = ''
                        if self._division_percent < 1:
                            suffix += '{}_newest'.format(self._division_percent)
                        authors_features = self._get_features(source_id_source_element_dict,
                                                              source_id_target_elements_dict,
                                                              suffix, targeted_fields_dict, aggregated_functions)
                        self._add_suffix_to_author_features(authors_features, suffix)
                        for feature in authors_features:
                            key = ','.join([feature.author_guid, feature.attribute_name])
                            if key not in seen_author_features:
                                total_authors_features.append(feature)
                                seen_author_features.add(key)
                        if len(total_authors_features) > self._max_objects_save:
                            self._db.add_author_features_fast(total_authors_features)
                            total_authors_features = []

        self._db.add_author_features_fast(total_authors_features)

    """ Author account properties """

    @fraction_results_decorator
    def posts_date_diff(self, **kwargs):
        posts_dates = self._get_posts_field('date', **kwargs)
        posts_dates = self._filter_empty_dates(posts_dates)
        return [(max(posts_dates) - min(posts_dates)).days] if posts_dates != [] else [0.0]

    def _filter_empty_dates(self, posts_dates):
        posts_dates = np.array([post_date for post_date in posts_dates if isinstance(post_date, datetime.datetime)])
        return posts_dates

    @fraction_results_decorator
    def author_screen_name_length(self, **kwargs):
        return list(map(len, self._get_authors_field('author_screen_name', **kwargs)))

    """ Posts properties """

    @fraction_results_decorator
    def posts_age(self, **kwargs):
        if isinstance(kwargs['author'], Claim):
            start_datetime = kwargs['author'].verdict_date
        elif hasattr(kwargs['author'], 'created_at'):
            start_date = parser.parse((getattr(kwargs['author'], 'created_at'))).date()
            start_datetime = datetime.datetime.fromordinal(start_date.toordinal())
        else:
            raise ValueError('{} does not have creation date'.format(type(kwargs['author'])))

        posts_dates = np.array(self._get_posts_field('date', **kwargs))
        posts_dates = self._filter_empty_dates(posts_dates)
        return [delta.days for delta in posts_dates - start_datetime]

    @fraction_results_decorator
    def num_of_retweets(self, **kwargs):
        return self._base_post_aggregated('retweet_count', **kwargs)

    @fraction_results_decorator
    def posts_num_of_favorites(self, **kwargs):
        return self._base_post_aggregated('favorite_count', **kwargs)

    @fraction_results_decorator
    def authors_age_diff(self, **kwargs):
        create_at_fields = self._get_authors_field('created_at', **kwargs)
        create_at_fields = [author_date for author_date in create_at_fields if isinstance(author_date, str)]
        author_creation_dates = [parser.parse(create_at).date() for create_at in create_at_fields]
        return [(max(author_creation_dates) - min(author_creation_dates)).days] if author_creation_dates != [] else [
            0.0]

    """ Followers """

    @fraction_results_decorator
    def num_of_followers(self, **kwargs):
        return self._base_author_aggregated('followers_count', **kwargs)

    """ Friends """

    @fraction_results_decorator
    def num_of_friends(self, **kwargs):
        return self._base_author_aggregated('friends_count', **kwargs)

    """ Statuses """

    @fraction_results_decorator
    def num_of_statuses(self, **kwargs):
        return self._base_author_aggregated('statuses_count', **kwargs)

    """ Favorites """

    @fraction_results_decorator
    def num_of_favorites(self, **kwargs):
        return self._base_author_aggregated('favourites_count', **kwargs)

    """ Listed """

    @fraction_results_decorator
    def num_of_listed_count(self, **kwargs):
        return self._base_author_aggregated('listed_count', **kwargs)

    """ Protected """

    @fraction_results_decorator
    def num_of_protected(self, **kwargs):
        return self._base_author_aggregated('protected', **kwargs)

    """ Verified """

    @fraction_results_decorator
    def num_of_verified(self, **kwargs):
        array = np.array(self._get_authors_field('verified', **kwargs), dtype=int)
        return array if list(array) != [] else np.zeros(1)

    """ Ratios """

    @fraction_results_decorator
    def friends_followers_ratio(self, **kwargs):
        friends_counts = np.array(self._get_authors_field('friends_count', **kwargs), dtype=float)
        followers_counts = np.array(self._get_authors_field('followers_count', **kwargs), dtype=float)
        result = np.divide(friends_counts, followers_counts)
        if list(result) == []:
            return [0.0]
        result[result == np.inf] = -1
        return result

    def _base_post_aggregated(self, field_name, **kwargs):
        return self._get_posts_field(field_name, **kwargs)

    @contains_posts_decorator
    def _get_posts_field(self, field_name, **kwargs):
        posts = kwargs['posts']
        results = [getattr(p, field_name, 0.0) for p in posts]
        return [field if field else 0.0 for field in results]

    def _base_author_aggregated(self, filed_name, **kwargs):
        return self._get_authors_field(filed_name, **kwargs)

    @contains_posts_decorator
    def _get_authors_field(self, filed_name, **kwargs):
        key = self._db.get_entity_key(kwargs['author'])
        key = key if key else str(kwargs['author'])
        if key in self._source_authors_dict:
            authors = self._source_authors_dict[key]
        else:
            if 'authors' in kwargs and kwargs['authors']:
                author_guids = set([a.author_guid for a in kwargs['authors']])
            else:
                author_guids = set([p.author_guid for p in kwargs['posts']])
            # self._source_authors_dict[key] = self.authors_df.loc[author_guids]
            self._source_authors_dict[key] = self.authors_df.loc[self.authors_df.index.intersection(author_guids)]
        # author_fields = map(lambda a: getattr(a, filed_name), authors)
        author_fields = self._source_authors_dict[key][filed_name].dropna()
        return author_fields
