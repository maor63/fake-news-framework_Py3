# Created by jorgeaug at 30/06/2016
from configuration.config_class import getConfig
import datetime
from .base_feature_generator import BaseFeatureGenerator


class BoostScoresFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        if 'authors' in list(kwargs.keys()) and 'posts' in list(kwargs.keys()):
            self.authors = kwargs['authors']
            self.author_guid_posts_dict = kwargs['posts']
            self._db = db
            self.config_parser = getConfig()
            start_date = self.config_parser.get("DEFAULT", "start_date").strip("date('')")
            self._window_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            self._window_size = datetime.timedelta(
                seconds=int(self.config_parser.get("DEFAULT", "window_analyze_size_in_sec")))
            # self._window_end = self._window_start + self._window_size
            all_boost_stats = self._db.get_all_authors_boost_stats()
            if all_boost_stats:
                self.boost_scores = {v.author_guid: v for v in all_boost_stats}
            else:
                self.boost_scores = {}
        else:
            raise Exception('Author object was not passed as parameter')

    def cleanUp(self):
        pass

    def boost_score_avg(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.boost_scores:
                return self.boost_scores[author.author_guid].scores_avg
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')

    def boost_score_std_dev(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.boost_scores:
                return self.boost_scores[author.author_guid].scores_std
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')

    def boost_score_sum(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.boost_scores:
                return self.boost_scores[author.author_guid].scores_sum
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')

    def boosting_timeslots_count(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.boost_scores:
                return self.boost_scores[author.author_guid].boosting_timeslots_participation_count
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')

    def count_authors_sharing_boosted_posts(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.boost_scores:
                return self.boost_scores[author.author_guid].count_of_authors_sharing_boosted_posts
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')

    def number_pointed_posts(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.boost_scores:
                return self.boost_scores[author.author_guid].num_of_pointed_posts
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')

    def number_pointers(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.boost_scores:
                return self.boost_scores[author.author_guid].num_of_pointers
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')
