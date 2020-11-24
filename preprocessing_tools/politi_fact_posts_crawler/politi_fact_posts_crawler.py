import json

import dateutil
import dateutil.parser
from bs4 import BeautifulSoup

from commons.commons import *
from preprocessing_tools.post_importer import PostImporter


class PolitiFactPostsCrawler(PostImporter):
    def __init__(self, db):
        # politifact posts title are biased, dont use them as features
        PostImporter.__init__(self, db)
        self._domain = "PolitiFact"
        self._subjects = self._actions = self._config_parser.eval(self.__class__.__name__, "subjects")
        self._posts_per_subject = self._actions = self._config_parser.eval(self.__class__.__name__, "posts_per_subject")
        self._post_types = self._actions = self._config_parser.eval(self.__class__.__name__, "post_types")
        self._author_classify_dict = {}
        self._author_prop_dict = {}
        self._post_type_dict = {}

    def execute(self, window_start):
        for subject in self._subjects:
            query = self._build_qury_for_subject(subject)
            posts_json = self.retrive_posts_json(query)
            self.extract_posts(posts_json)
            if self._listdic is None:
                return None
            self.insertPostsIntoDB()
            self._db.insert_or_update_authors_from_posts(self._domain, self._author_classify_dict,
                                                         self._author_prop_dict)

    def extract_posts(self, posts_json):
        for post in posts_json:
            post = self._parse_post_from_json(post)
            if post is None:
                continue
            self._listdic.append(post)

    def retrive_posts_json(self, query):
        req = urllib.request.Request(query, headers={'User-Agent': "Magic Browser"})
        respons = urllib.request.urlopen(req)
        posts_json = json.load(respons)
        return posts_json

    def _build_qury_for_subject(self, subject):
        query = "http://www.politifact.com/api/statements/truth-o-meter/subjects/" + subject + "/json/?n=" + str(
            self._posts_per_subject)
        return query

    def get_claim_by_id(self, claim_id):
        query = "http://www.politifact.com/api/statements/truth-o-meter/detail/" + claim_id + "/json/"
        retrieved_posts = self.retrive_posts_json(query)
        return retrieved_posts

    def _parse_post_from_json(self, post_json):
        post_dict = {}

        post_dict["url"] = "http://www.politifact.com" + post_json['statement_url']
        d = dateutil.parser.parse(post_json['statement_date']).strftime('%Y-%m-%d %H:%M:%S')
        post_dict["date"] = d
        author_name = post_dict["url"]

        post_guid = compute_post_guid(post_dict["url"], author_name, post_dict["date"]).replace('-', '')

        post_dict["title"] = post_json['ruling_headline']
        post_dict["created_at"] = post_dict["date"]
        post_dict["content"] = self._clean_string(post_json["statement"])
        post_dict["domain"] = self._domain
        post_dict["guid"] = post_guid
        post_dict["post_guid"] = post_guid
        post_dict["post_id"] = post_guid
        post_dict["author_guid"] = post_guid
        post_dict["author"] = author_name
        post_dict["author_osn_id"] = post_dict["author_guid"]
        post_dict["references"] = ""

        post_type = post_json["ruling"]["ruling_slug"]
        if post_type in self._post_types:
            post_type = self._post_types[post_type]
        else:
            return None

        post_dict["post_type"] = post_type
        self._add_property_to_author_prop_dict(post_guid, "author_name", author_name)
        self._add_property_to_author_prop_dict(post_guid, "author_guid", post_guid)
        self._add_property_to_author_prop_dict(post_guid, "author_screen_name", author_name)
        self._add_property_to_author_prop_dict(post_guid, "author_osn_id", post_guid)
        self._add_property_to_author_prop_dict(post_guid, "author_type", post_dict["post_type"])
        self._add_property_to_author_prop_dict(post_guid, "author_sub_type", post_json["ruling"]["ruling_slug"])

        return post_dict

    def _clean_string(self, string):
        clean_str = ''
        for word in string.split(' '):
            clean_word = self._clean_words(word)
            if clean_word != '':
                clean_str = clean_str + ' ' + clean_word
        return clean_str[1:len(clean_str)]

    def _clean_words(self, word):
        soup = BeautifulSoup(word, "lxml")
        not_html_tags = soup.get_text()
        return re.sub('[^A-Za-z0-9]+', '', not_html_tags)

    def _add_property_to_author_prop_dict(self, author_guid, key, value):
        if value is not None:
            if author_guid not in self._author_prop_dict:
                self._author_prop_dict[author_guid] = {}
            self._author_prop_dict[author_guid][key] = value

    def _post_type_by_dict(self, post_type):
        return self._post_types[post_type]
