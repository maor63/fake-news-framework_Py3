

import csv
import json
from os import listdir

from bs4 import BeautifulSoup

from DB.schema_definition import *
from preprocessing_tools.abstract_controller import AbstractController
from preprocessing_tools.json_importer.json_importer_parent import JSON_Importer_Parent


class JSON_Importer(JSON_Importer_Parent):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._actions = self._config_parser.eval(self.__class__.__name__, "actions")
        self._json_path = self._config_parser.eval(self.__class__.__name__, "json_path")
        self._path_bad_actor_csv_path = self._config_parser.eval(self.__class__.__name__, "path_bad_actor_csv")
        self._author_type = str(self._config_parser.get(self.__class__.__name__, "author_type"))
        self._author_dict = {}
        self._authors = []
        self._posts = []

    def mark_author_type_from_csv(self, window_start=None):
        all_sub_files = listdir(self._path_bad_actor_csv_path)
        for csv_file in all_sub_files:
            self._mark_author_type_from_file(csv_file)

    def _mark_author_type_from_file(self, csv_file):
        if os.path.isfile(self._path_bad_actor_csv_path + csv_file):
            f = open(self._path_bad_actor_csv_path + csv_file)
            reader = csv.DictReader(f, delimiter=',')
            bad_author_names = [row['id'] for row in reader]

            self._db.update_authors_type_by_author_names(bad_author_names, self._author_type)
        else:
            print("There isn't bad_actor CSV")

    def read_from_files(self):
        json_files = listdir(self._json_path)
        i = 1
        for json_file in json_files:
            msg = "\rAnalyzing json {0}/{1} ".format(str(i), len(json_files))
            print(msg, end="")
            i += 1
            if os.path.isfile(self._json_path + json_file):
                self._parse_json(json_file)
                if i % 1000 == 0:
                    print("Save after 500 files")
                    self._db.add_authors(self._authors)
                    self._db.addPosts(self._posts)
                    self._authors = []
                    self._posts = []
        self._db.add_authors(self._authors)
        self._db.addPosts(self._posts)

    def _parse_json(self, curr_file):
        raw_json_data = self._parse_json_files(curr_file)
        if 'Depth' in list(raw_json_data.keys()):
            return
        json_data_str = raw_json_data[list(raw_json_data.keys())[0]]
        json_data_ready_for_parse = json.loads(json_data_str)
        if list(raw_json_data.keys())[0].find('user_id') != -1:
            self._parse_author(json_data_ready_for_parse)
        else:
            for tweet in json_data_ready_for_parse:
                self._parse_post(tweet)

    def _parse_post(self, json_data_ready_for_parse):
        post = Post()
        post.post_id = str(json_data_ready_for_parse['id'])
        post.author = str(json_data_ready_for_parse['user']['screen_name'])
        post.author_guid = str(
            compute_author_guid_by_author_name(json_data_ready_for_parse['user']['screen_name']).replace('-', ''))
        if 'cards' in json_data_ready_for_parse and 'photos' in json_data_ready_for_parse['cards']:
            post.title = str(json_data_ready_for_parse['cards']['photos'][0]['title'])
        content = BeautifulSoup(json_data_ready_for_parse['text'], "lxml")
        post.url = str(self._social_network_url + "{0}/status/{1}".format(post.author, post.post_id))
        date_str = json_data_ready_for_parse['created_at'][0:19].replace("/", "-")
        post.date = str_to_date(date_str)
        post.guid = str(compute_post_guid(post.url, post.author, date_str))
        post.content = str(content.text)
        post.domain = self._domain
        post.post_osn_guid = str(post.guid)
        post.post_osn_id = str(json_data_ready_for_parse['id'])
        post.retweet_count = json_data_ready_for_parse['retweet_count']
        post.favorite_count = json_data_ready_for_parse['favorite_count']
        post.created_at = post.date
        Post.xml_importer_insertion_date = self._window_start
        self._posts.append(post)
        self._parse_author(json_data_ready_for_parse['user'])

    def _parse_author(self, json_data_ready_for_parse):
        if json_data_ready_for_parse['screen_name'] not in self._author_dict:
            author = Author()
            author.name = str(json_data_ready_for_parse['screen_name'])
            author.author_guid = str(
                compute_author_guid_by_author_name(json_data_ready_for_parse['screen_name']).replace('-', ''))
            author.author_screen_name = str(json_data_ready_for_parse['screen_name'])
            author.domain = self._domain
            author.author_osn_id = str(json_data_ready_for_parse['id'])
            author.created_at = str(json_data_ready_for_parse['created_at'])
            author.default_profile = str(json_data_ready_for_parse['default_profile'])
            author.author_full_name = str(json_data_ready_for_parse['name'])
            author.favourites_count = str(json_data_ready_for_parse['favourites_count'])
            author.followers_count = json_data_ready_for_parse['followers_count']
            author.friends_count = json_data_ready_for_parse['friends_count']
            author.url = str(self._social_network_url + "{0}".format(author.name))
            author.default_profile_image = str(json_data_ready_for_parse['default_profile_image'])
            author.language = str(json_data_ready_for_parse['lang'])
            description = BeautifulSoup(json_data_ready_for_parse['description'], "lxml")
            author.description = str(description.text)
            author.listed_count = json_data_ready_for_parse['listed_count']
            author.location = str(json_data_ready_for_parse['location'])
            author.profile_background_color = str(json_data_ready_for_parse['profile_background_color'])
            author.profile_background_tile = str(json_data_ready_for_parse['profile_background_tile'])
            author.profile_sidebar_fill_color = str(json_data_ready_for_parse['profile_sidebar_fill_color'])
            author.profile_image_url = str(json_data_ready_for_parse['profile_image_url'])
            author.profile_link_color = str(json_data_ready_for_parse['profile_link_color'])
            author.profile_text_color = str(json_data_ready_for_parse['profile_text_color'])
            author.geo_enabled = str(json_data_ready_for_parse['geo_enabled'])
            author.verified = json_data_ready_for_parse['verified']
            author.protected = json_data_ready_for_parse['protected']
            author.statuses_count = (json_data_ready_for_parse['statuses_count'])
            author.contributors_enabled = str(json_data_ready_for_parse['contributors_enabled'])
            author.time_zone = str(json_data_ready_for_parse['time_zone'])
            author.notifications = str(json_data_ready_for_parse['notifications'])
            author.utc_offset = str(json_data_ready_for_parse['utc_offset'])
            author.xml_importer_insertion_date = self._window_start
            self._authors.append(author)
            self._author_dict[author.name] = author.name
