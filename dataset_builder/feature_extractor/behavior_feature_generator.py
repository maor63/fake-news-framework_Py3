# Created by jorgeaug at 30/06/2016
import math
import pandas as pd
from dateutil import parser
import datetime
from .base_feature_generator import BaseFeatureGenerator
from commons.consts import Social_Networks
from commons.commons import *


class BehaviorFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._retweet_count = db.get_retweet_count()
        self._retweets = pd.DataFrame(list(db.get_retweets().items()), columns=['post_id', 'content'])
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._author_dict = {}

    def execute(self, window_start=None):
        self._author_dict = self._db.get_author_dictionary()
        self._get_author_features_using_args(self._targeted_fields)

    def average_minutes_between_posts(self, **kwargs):
        if 'posts' in list(kwargs.keys()):
            posts = kwargs['posts']
            data_frame = self.convert_posts_to_dataframe(posts)
            time_series = data_frame['date']
            time_series = time_series.sort_values(ascending=False)
            if len(time_series) > 1:
                diff = time_series - time_series.shift()
                mean = diff.mean().total_seconds() / 60
                if not math.isnan(mean):
                    avg_minutes_between_posts = math.fabs(mean)
                else:
                    avg_minutes_between_posts = None

            else:
                avg_minutes_between_posts = 0

            return avg_minutes_between_posts
        else:
            return 0

    def average_posts_per_day_active_days(self, **kwargs):
        if 'posts' in list(kwargs.keys()):
            posts = kwargs['posts']
            data_frame = self.convert_posts_to_dataframe(posts)
            data_frame['date'] = data_frame['date'].map(lambda x: x.strftime('%Y-%m-%d'))
            grouped_data = data_frame.groupby('date')
            daily_count = grouped_data.date.apply(pd.value_counts)
            if len(daily_count) == 0:
                return 0
            avg_posts_per_active_day = daily_count.mean()
            return avg_posts_per_active_day
        else:
            return 0

    def average_posts_per_day_total(self, **kwargs):
        # if 'author' in kwargs.keys():
        #     author = kwargs['author']
        post_dats = self._get_posts_field('date', **kwargs)
        time_delta = (max(post_dats) - min(post_dats)).days
        # return self._calculate_average_posts_per_day_total_within_twitter(author)
        return 0 if time_delta == 0 else float(len(kwargs['posts'])) / time_delta
        # elif self._targeted_social_network == Social_Networks.TUMBLR:
        #     return self._calculate_average_posts_per_day_total_within_tumblr(author)

    def retweet_count(self, **kwargs):
        if 'author' in kwargs and 'posts' in kwargs:
            return self._count_retweets(kwargs['posts'])
        else:
            return 0

    def _count_retweets(self, posts):
        return sum([1 for p in posts if "RT @" in p.content])

    def average_retweets(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            posts = kwargs['posts']
            if len(posts) > 0:
                return self._count_retweets(kwargs['posts']) / float(len(posts))
            else:
                return 0.0

    def received_retweets_count(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author_screen_names = self._get_authors_field('author_screen_name', **kwargs)
            retweets_count = 0
            for author_screen_name in author_screen_names:
                screen_name = str('RT @' + author_screen_name)
                retweets_count += len(self._retweets[self._retweets['content'].str.contains(screen_name)].index)
            return retweets_count

    def _calculate_average_posts_per_day_total_within_twitter(self, author):
        if author.created_at is not None and author.statuses_count is not None:
            account_creation_date = parser.parse(author.created_at).date()
            today_date = datetime.date.today()
            delta = today_date - account_creation_date
            total_days = float(delta.days)

            total_posts = float(author.statuses_count)
            if total_posts > 0 and total_days > 0:
                return total_posts / total_days
            else:
                return 0

    def _calculate_average_posts_per_day_total_within_tumblr(self, author):
        if author.created_at is not None:
            author_creation_date = author.created_at
            account_creation_date = str_to_date(author_creation_date)
            account_creation_date = account_creation_date.date()

            today_date = datetime.date.today()
            delta = today_date - account_creation_date
            total_days = float(delta.days)

            author_guid = author.author_guid

            posts = self._db.get_posts_by_author_guid(author_guid)
            total_posts = len(posts)
            # total_posts = float(author.statuses_count)
            if total_posts > 0 and total_days > 0:
                return total_posts / total_days

    def _get_posts_field(self, field_name, **kwargs):
        posts = kwargs['posts']
        return [getattr(p, field_name) for p in posts]

    def _base_author_aggregated(self, filed_name, **kwargs):
        return self._get_authors_field(filed_name, **kwargs)

    def _get_authors_field(self, filed_name, **kwargs):
        author_guids = set([p.author_guid for p in kwargs['posts']])
        authors = [self._author_dict[a_guid] for a_guid in author_guids if a_guid in self._author_dict]
        return [getattr(a, filed_name) for a in authors]
