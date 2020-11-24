from configuration.config_class import getConfig
import datetime
from .base_feature_generator import BaseFeatureGenerator
import logging

class KeyAuthorScoreFeatureGenerator(BaseFeatureGenerator):
    '''
    Extracts the SumTFIDF and MaxTFIDF from the export_key_Authors view.
    Notes:
        * This view is created by the KeyAuthorsModel. Therefore, for this module to work properly,
          KeyAuthorsModel needs to be executed first
        * Not all authors will have max\sumtfidf score
    '''
    def __init__(self, db, **kwargs):
        self._db = db
        self.config_parser = getConfig()

        start_date = self.config_parser.get("DEFAULT", "start_date").strip("date('')")
        self._window_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self._window_size = datetime.timedelta(
            seconds=int(self.config_parser.get("DEFAULT", "window_analyze_size_in_sec")))
        self._window_end = self._window_start + self._window_size

        if 'authors' in kwargs and 'posts' in kwargs:
            self.authors = kwargs['authors']
            self.author_guid_posts_dict = kwargs['posts']
        else:
            raise Exception('Author object was not passed as parameter')

        if not self._db.is_export_key_authors_view_exist():  # the required view,export_key_Authors, doesn't exist.
            logging.error("Cannot initiate KeyAuthorScoreFeatureGenerator as the export_key_authors view does not appear in the db")
            self.module_enabled = False
        else:
            self.module_enabled = True
            self.sum_tfidf_dict = db.get_sum_tfidf_scores()
            self.max_tfidf_dict = db.get_max_tfidf_scores()

    def cleanUp(self):
        pass

    def sum_tfidf(self, **kwargs):
        if not self.module_enabled:
            return None
        elif 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.sum_tfidf_dict:
                return self.sum_tfidf_dict[author.author_guid]
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')

    def max_tfidf(self, **kwargs):
        if not self.module_enabled:
            return None
        elif 'author' in list(kwargs.keys()):
            author = kwargs['author']
            if author.author_guid in self.max_tfidf_dict:
                return self.max_tfidf_dict[author.author_guid]
            else:
                return None
        else:
            raise Exception('Author object was not passed as parameter')