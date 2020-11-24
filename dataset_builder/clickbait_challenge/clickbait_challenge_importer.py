'''
@author: Aviad Elyashar aviade@post.bgu.ac.il
'''

from preprocessing_tools.json_importer.json_importer_parent import JSON_Importer_Parent
import json
from DB.schema_definition import Post, Author, Target_Article ,Target_Article_Item
from commons.commons import *
import dateutil

__author__ = "Aviad Elyashar"


class Clickbait_Challenge_Importer(JSON_Importer_Parent):
    def __init__(self, db):
        JSON_Importer_Parent.__init__(self, db)
        self._actions = self._config_parser.eval(self.__class__.__name__, "actions")
        self._json_path = self._config_parser.eval(self.__class__.__name__, "json_path")
        self._records_file_name = self._config_parser.eval(self.__class__.__name__, "records_file_name")
        self._truth_file_name = self._config_parser.eval(self.__class__.__name__, "truth_file_name")
        self._dataset_type = self._config_parser.eval(self.__class__.__name__, "dataset_type")

    def read_from_files(self):
        self._read_truth_file()
        self._read_records_file_and_add_to_db()


    def _read_records_file_and_add_to_db(self):
        posts = []
        authors = []
        target_articles = []
        target_article_items = []
        author_features = []
        with open(self._json_path + self._records_file_name) as file:
                for line in file:
                    json_content = json.loads(line)
                    post = self._parse_post(json_content)
                    posts.append(post)

                    author = self._parse_author(json_content)
                    author_guid = author.author_guid
                    print('\r Author {0} was parsed! '.format(author.author_guid), end="")
                    authors.append(author)

                    target_article = self._parse_target_article(json_content, author.author_guid)
                    target_articles.append(target_article)

                    target_article_items = self._parse_target_article_items(json_content, target_article_items, author_guid)

                    if self._dataset_type is not None:
                        set_affiliation_author_feature = self._create_set_affiliation_author_feature(author)
                        author_features.append(set_affiliation_author_feature)

                file.close()
        self._db.addPosts(posts)
        self._db.addAuthors(authors)
        self._db.add_author_features(author_features)
        self._db.add_target_articles(target_articles)
        self._db.addPosts(target_article_items)


    def _parse_json_files(self, currfolder):
        raw_json_file = open(self._json_path + currfolder)
        raw_json_str = raw_json_file.read()
        raw_json_data = json.loads(raw_json_str)
        return raw_json_data

    def _parse_post(self, json_content):
        post = Post()
        content = json_content['postText'][0]
        post.content = content

        str_post_timestamp = json_content["postTimestamp"]
        post.created_at = str(str_post_timestamp)

        # post_timestamp = dateutil.parser.parse(str_post_timestamp)
        # str_post_date = date_to_str(post_timestamp)

        post_timestamp, str_post_date = self._get_str_and_date_formats(str_post_timestamp)
        post.date = post_timestamp

        post_media = json_content["postMedia"]
        if len(post_media) > 0:
            post.media_path = post_media[0]
        else:
            post.media_path = None

        post_id = json_content["id"]
        post.post_id = post_id

        post.author = post_id

        post_guid = compute_post_guid(self._social_network_url, post_id, str_post_date)
        post.guid = post_guid
        post.domain = self._domain
        post.author_guid = post_guid
        post.post_osn_guid = post_guid
        return post

    def _parse_author(self, json_content):
        author = Author()

        post_id = json_content["id"]
        author.name = post_id
        author.author_screen_name = post_id
        author.author_osn_id = post_id
        author.domain = self._domain

        str_post_timestamp = json_content["postTimestamp"]
        post_timestamp, str_post_date = self._get_str_and_date_formats(str_post_timestamp)
        author.created_at = str(str_post_date)

        post_guid = compute_post_guid(self._social_network_url, post_id, str_post_date)
        author.author_guid = post_guid

        if post_id in self._post_id_targeted_class_dict:
            targeted_class = self._post_id_targeted_class_dict[post_id]
            author.author_type = targeted_class

        post_media = json_content["postMedia"]
        if len(post_media) > 0:
            author.media_path = post_media[0]
        else:
            author.media_path = None

        author.notifications = self._dataset_type

        return author

    def _create_set_affiliation_author_feature(self, author):
        attribute_name = str("set_affiliation")
        attribute_value = self._dataset_type

        author_guid = author.author_guid

        set_affiliation_author_feature = self._db.create_author_feature(author_guid, attribute_name, attribute_value)
        return set_affiliation_author_feature


    def _get_str_and_date_formats(self, str_timestamp):
        date_timestamp = dateutil.parser.parse(str_timestamp)
        str_post_date = date_to_str(date_timestamp)
        return date_timestamp, str_post_date

    def _read_truth_file(self):
        self._post_id_targeted_class_dict = {}
        if self._truth_file_name is not None:
            with open(self._json_path + self._truth_file_name) as file:
                    for line in file:
                        json_content = json.loads(line)
                        post_id = json_content["id"]
                        truth_class = json_content["truthClass"]
                        self._post_id_targeted_class_dict[post_id] = truth_class

    def _parse_target_article(self, json_content, author_guid):
        target_article = Target_Article()

        post_id = json_content["id"]
        target_article.post_id = post_id

        target_article.author_guid = author_guid

        title = json_content["targetTitle"]
        target_article.title = title

        description = json_content["targetDescription"]
        target_article.description = description

        keywords = json_content["targetKeywords"]
        target_article.keywords = keywords

        return target_article

    def _parse_target_article_items(self, json_content, target_article_items, author_guid):

        post_id = json_content["id"]

        target_paragraphs = json_content["targetParagraphs"]
        item_type = 'paragraph'
        target_article_items = self._fill_target_article_items_by_type(post_id, author_guid, target_paragraphs, item_type, target_article_items)

        target_captions = json_content["targetCaptions"]
        item_type = 'caption'
        target_article_items = self._fill_target_article_items_by_type(post_id, author_guid, target_captions, item_type,
                                                                       target_article_items)

        return target_article_items

    def _create_target_article_item(self, post_id, author_guid, item_type, i, paragraph):
        target_article_item = Target_Article_Item()

        target_article_item.post_id = post_id
        target_article_item.author_guid = author_guid
        target_article_item.type = item_type
        target_article_item.item_number = i
        target_article_item.content = paragraph

        return target_article_item

    def _fill_target_article_items_by_type(self, post_id, author_guid, target_paragraphs, item_type, target_article_items):
        i = 0
        for paragraph in target_paragraphs:
            target_article_item = self._create_target_article_item(post_id, author_guid,  item_type, i, paragraph)
            target_article_items.append(target_article_item)
            i += 1
        return target_article_items
