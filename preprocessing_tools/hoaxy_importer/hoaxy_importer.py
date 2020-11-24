
from bs4 import BeautifulSoup

from commons.commons import *
from preprocessing_tools.post_importer import PostImporter
import urllib.request, urllib.error, urllib.parse
import json
import datetime, dateutil.parser

class Hoaxy_Importer(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        self._past_hours = self._config_parser.eval(self.__class__.__name__, "past_hours")
        self._author_name_as_domain = self._config_parser.eval(self.__class__.__name__, "author_name_as_domain")
        self._author_classify_dict = {}
        self._author_prop_dict = {}
        # self._start_date = self._config_parser.eval(self.__class__.__name__, "start_date")
        # self._end_date = self._config_parser.eval(self.__class__.__name__, "end_date")

    def execute(self, window_start):
        query = self._query_builder()
        req = urllib.request.Request(query)
        req.add_header("X-Mashape-Key", "iWkyJSfqvomshcnQZqqFRPl7EEXsp1548lDjsn1kSchUjB9CNe")
        req.add_header("Accept", "application/json")
        respons = urllib.request.urlopen(req)
        self._listdic = self._create_posts_list(respons)
        if self._listdic is None:
            return None
        self.insertPostsIntoDB()
        self._db.insert_or_update_authors_from_posts(self._domain, self._author_classify_dict,
                                                     self._author_prop_dict)

    def _query_builder(self):
        query = "https://api-hoaxy.p.mashape.com/latest-articles?past_hours="
        query = query + str(self._past_hours)
        return query

    def _create_posts_list(self, result_json):
        posts = []
        parsed = json.load(result_json)
        if parsed['status'] == 'No result error':
            print (parsed['status'], end="")
            return None
        articles = parsed['articles']
        for article in articles:
            post_dict = {}

            post_dict["url"] = article['canonical_url']
            d = dateutil.parser.parse(article['date_published']).strftime('%Y-%m-%d %H:%M:%S')
            post_dict["date"] = d
            author_name = post_dict["url"]

            post_guid = compute_post_guid(post_dict["url"], author_name, post_dict["date"]).replace('-', '')
            if not self._author_name_as_domain:
                author_name += "/" + post_guid

            post_dict["title"] = article['title']
            post_dict["created_at"] = post_dict["date"]
            post_dict["content"] = self._get_post_text(post_dict["url"])
            if post_dict["content"] is None:
                continue
            post_dict["domain"] = str(self._domain)
            post_dict["guid"] = post_guid
            post_dict["post_guid"] = post_guid
            post_dict["post_id"] = post_guid
            post_dict["author_guid"] = post_guid
            post_dict["author"] = author_name
            post_dict["author_osn_id"] = post_dict["author_guid"]
            post_dict["references"] = ""

            posts.append(post_dict)

            self._add_property_to_author_prop_dict(post_guid, "author_name", author_name)
            self._add_property_to_author_prop_dict(post_guid, "author_guid", post_guid)
            self._add_property_to_author_prop_dict(post_guid, "author_screen_name", author_name)
            self._add_property_to_author_prop_dict(post_guid, "author_osn_id", post_guid)

        return posts

    def _get_post_text(self, url):
        try:
            html_doc = urllib.request.urlopen(url)
            self.logger.info("checkin post "+url)
            print(url, end="")
            soup = BeautifulSoup(html_doc, 'html.parser')
            content = soup.find_all('p')
            # title = soup.find_all('h1')
            all_content = ""
            for p in content:
                all_content += p.text.replace('\r', '').replace('\n', '')
            return all_content
        except:
            return None

    def _add_property_to_author_prop_dict(self, author_guid, key, value):
        if value is not None:
            if author_guid not in self._author_prop_dict:
                self._author_prop_dict[author_guid] = {}
            self._author_prop_dict[author_guid][key] = value

