# Created by jorgeaug at 30/06/2016
from .base_feature_generator import BaseFeatureGenerator


class SyntaxFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        super(SyntaxFeatureGenerator, self).__init__(db, **{'authors': [], 'posts': {}})
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")

    def execute(self, window_start=None):
        self._get_author_features_using_args(self._targeted_fields)

    def average_hashtags(self, **kwargs):
        filter_fn = lambda token: '#' in token
        return self._average_apperence_in_posts(kwargs['posts'], filter_fn)

    def average_links(self, **kwargs):
        filter_fn = lambda token: any(s in token for s in ('http://', 'https://'))
        return self._average_apperence_in_posts(kwargs['posts'], filter_fn)

    def average_user_mentions(self, **kwargs):
        filter_fn = lambda token: '@' in token
        return self._average_apperence_in_posts(kwargs['posts'], filter_fn)

    def average_post_lenth(self, **kwargs):
        filter_fn = lambda token: token
        return self._average_apperence_in_posts(kwargs['posts'], filter_fn)

    def _average_apperence_in_posts(self, posts, filter_by):
        appearances = 0.0
        number_posts = len(posts)
        for post in posts:
            if post.content is not None:
                content = post.content.lower()
                appearances += sum(1 for word in content.split(" ") if filter_by(word))
        return appearances / number_posts
