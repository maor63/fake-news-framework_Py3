import calendar
from os import listdir

from datetime import datetime

from DB.schema_definition import Author, Post, DB
from commons.commons import *
from preprocessing_tools.abstract_controller import AbstractController
import json


class BuzzFeedPolitiFactImporter(AbstractController):
    # The data set is retrieved from "https://github.com/KaiDMML/FakeNewsNet"
    def __init__(self, db):
        super(BuzzFeedPolitiFactImporter, self).__init__(db)
        self._fake_news_path = self._config_parser.eval(self.__class__.__name__, "fake_news_path")
        self._real_news_path = self._config_parser.eval(self.__class__.__name__, "real_news_path")

    def execute(self, window_start=None):
        self.add_author_and_posts(self._fake_news_path, 'Fake')
        self.add_author_and_posts(self._real_news_path, 'Real')

    def add_author_and_posts(self, news_path, post_type):
        all_sub_files = listdir(news_path)
        authors = []
        posts = []
        for json_file in all_sub_files:
            data = self._parse_json(news_path + json_file)
            if data == "":
                print(("JSON {0} is empty".format(json_file)))
                continue
            if 'source' not in data:
                print(("JSON {0} has no author".format(json_file)))
                continue
            author = self.extract_author(data)
            authors.append(author)
            post = self.extract_post(data, post_type)
            posts.append(post)
        assert (isinstance(self._db, DB))
        self._db.add_authors(authors)
        self._db.addPosts(posts)

    def extract_post(self, data, post_type):
        post = Post()
        if data['publish_date'] is None:
            publish_date_date = calendar.timegm(time.gmtime()) * 1000
        else:
            publish_date_date = data['publish_date']['$date']
        date_str = datetime.datetime.fromtimestamp(publish_date_date / 1000).strftime('%Y-%m-%d %H:%M:%S')
        post.post_id = compute_post_guid(data['url'], data['source'], date_str)
        post.guid = post.post_id
        post.author_guid = compute_author_guid_by_author_name(data['source'])
        post.author = str(data['source'])
        post.date = convert_str_to_unicode_datetime(date_str)
        post.title = str(data['title'])
        post.url = str(data['url'])
        post.source_url = str(data['source'])
        post.content = str(data['text'])
        post.tags = ','.join(data['keywords'])
        post.domain = self._domain
        post.post_type = post_type
        if 'description' not in data['meta_data']:
            post.description = ""
        else:
            post.description = str(data['meta_data']['description'])
        return post

    def extract_author(self, data):
        author = Author()
        author.name = str(data['source'])
        author.domain = 'BuzzFeed'
        author.author_guid = compute_author_guid_by_author_name(data['source'])
        author.author_screen_name = author.name
        author.author_full_name = author.name
        return author

    def _parse_json(self, json_path):
        raw_json_file = open(json_path)
        raw_json_str = raw_json_file.read()
        if raw_json_str == "":
            return ""
        raw_json_data = json.loads(raw_json_str)
        return raw_json_data
