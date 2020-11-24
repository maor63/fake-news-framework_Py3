# Created by Aviad Elyashar at 25/04/2017

import urllib.request, urllib.parse, urllib.error

from preprocessing_tools.Topic_Term_Manager import Topic_Term_Manager
from preprocessing_tools.post_importer import PostImporter
from gdelt_full_text_search_api.gdelt_full_text_search_api import GDLET_Full_Text_Search_API
from commons.commons import *
from io import StringIO
from bs4 import BeautifulSoup

class GDLET_News_Importer(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        self._author_name_as_domain = self._config_parser.eval(self.__class__.__name__, "author_name_as_domain")
        self._retrieve_news_by_keywords = self._config_parser.eval(self.__class__.__name__, "retrieve_news_by_keywords")
        self._num_of_top_terms = self._config_parser.eval(self.__class__.__name__, "num_of_top_terms")
        self._filter_sentences = self._config_parser.eval(self.__class__.__name__, "filter_sentences")
        self._topic_term_manager = Topic_Term_Manager(db)
        self._characters_to_add_to_unstemmed_words = ['e', 'able', 'al', 'ial' 'ion', 'ing', 'er', 'ies']

    def execute(self, window_start):

        self._gdlet_api = GDLET_Full_Text_Search_API(self._db)

        while True:
            if self._retrieve_news_by_keywords:
                results = self._gdlet_api.get_news_by_keywords()
                self._convert_and_save_results(results)
            else: # retrieve top 100 terms from topic_stats. You should run Autotopics Executor first!!!
                #self._retrieve_top_terms_from_best_topic_and_save()
                self._retrieve_news_by_top_terms_in_each_topic()


    def _create_posts_list_from_urls(self, url_date_dict):
        i = 1
        for url in url_date_dict:
            msg = "\r Url:{0} {1}/{2}".format(url, str(i), str(len(url_date_dict)))
            print(msg, end="")
            i += 1
            # Author as site url
            start = url.find("//") + 2
            end = url.find("/", start)
            author_name = url[start:end]
            #author_guid = compute_author_guid_by_author_name(author_name)#.replace('-', '')
            post_guid = str(compute_post_guid(url, author_name, url_date_dict[url]))#.replace('-', '')

            if not self._author_name_as_domain:
                author_name += "/" + post_guid
            try:
                html_doc = urllib.request.urlopen(url)
                soup = BeautifulSoup(html_doc, 'html.parser')
                content = soup.find_all('p')
                #title = soup.find_all('h1')
                all_content = ""
                for p in content:
                    all_content += p.text.replace('\r', '').replace('\n', '')

                if len(content) > 0:
                    post_dict = {}
                    post_dict["post_id"] = post_guid
                    title = soup.title
                    if title is not  None:
                        post_dict["title"] = str(soup.title.string)
                    else:
                        post_dict["title"] = title
                    post_dict["author"] = str(author_name)
                    post_dict["author_guid"] = post_guid

                    self._add_property_to_author_prop_dict(post_guid, "author_name", author_name)
                    self._add_property_to_author_prop_dict(post_guid, "author_guid", post_guid)
                    self._add_property_to_author_prop_dict(post_guid, "author_screen_name", author_name)
                    self._add_property_to_author_prop_dict(post_guid, "author_osn_id", post_guid)

                    post_dict["content"] = str(all_content)
                    #Lior Added - check blocked sites
                    if self._check_filter_sentences(post_dict["content"]) is True:
                        continue
                    post_dict["created_at"] = str(url_date_dict[url])
                    post_dict["date"] = post_dict["created_at"]
                    self._add_property_to_author_prop_dict(post_guid, "created_at", str(post_dict["created_at"]))
                    post_dict['guid'] = post_guid
                    post_dict["references"] = ""
                    post_dict["domain"] = str(self._domain)
                    post_dict["url"] = str(url)
                    self._add_property_to_author_prop_dict(post_guid, "url", str(url))

                    post_dict["author_osn_id"] = post_dict["author_guid"]

                    # self._author_classify_dict[author_name] = self._author_type
                    self._listdic.append(post_dict.copy())

            except IOError as e:
                logging.info("Problem with the url: {0}".format(url))
                error = e.strerror
                logging.info("errorno {0}: {1}".format(str(error.errno), error.strerror))
                continue
            except:
                e = sys.exc_info()[0]
                print("Unexpected error:", e.message)
                logging.info("Other Problem with the url: {0}".format(url))
                continue


    def _create_url_date_dict(self, results):
        url_date_dict = {}
        for result in results.readlines():
            raw_data = result.split(',')
            url = raw_data[2].replace('\n', '')
            url_date_dict[url] = "{0}-{1}-{2} {3}:{4}:{5}".format(raw_data[0][0:4], raw_data[0][4:6], raw_data[0][6:8],
                                                                  raw_data[0][8:10],
                                                                  raw_data[0][10:12], raw_data[0][12:14])
        return url_date_dict

    def _verify_results(self, original_results, term):
        data = str(original_results.read())
        content = StringIO()
        content.write(data)
        if len(content.getvalue()) == 0:
            for character in self._characters_to_add_to_unstemmed_words:
                new_term = ''
                new_term = term + character
                results = self._gdlet_api.get_news_by_keywords([new_term])
                new_data = str(results.read())
                content.write(new_data)
                if len(content.getvalue()) > 0:
                    content.seek(0)
                    return content
        return original_results

    def _convert_and_save_results(self, results):
        url_date_dict = self._create_url_date_dict(results)
        self._create_posts_list_from_urls(url_date_dict)
        self.insertPostsIntoDB()
        self._db.insert_or_update_authors_from_posts(self._domain, self._author_classify_dict,
                                                     self._author_prop_dict)

    def _retrieve_top_terms_from_best_topic_and_save(self):
        top_terms = self._topic_term_manager.get_term_from_db_with_most_posts()
        i = 0
        for term in top_terms:
            i += 1
            print("term: {0} {1}/{2}".format(term, str(i), str(len(top_terms))))
            results = self._gdlet_api.get_news_by_keywords([term])
            verified_results = self._verify_results(results, term)
            'if results is not empty'
            data = verified_results.read()
            if len(data) > 0:
                verified_results.seek(0)
                self._convert_and_save_results(verified_results)

    def _retrieve_news_by_top_terms_in_each_topic(self):
        topic_top_terms_dict = self._topic_term_manager.get_topic_top_terms_dictionary(self._num_of_top_terms)
        for topic, top_terms in topic_top_terms_dict.items():
            i = 0
            for term in top_terms:
                i += 1
                print("topic: {0} term: {1} {2}/{3}".format(topic, term, str(i), str(len(top_terms))))
                results = self._gdlet_api.get_news_by_keywords([term])
                verified_results = self._verify_results(results, term)
                'if results is not empty'
                data = verified_results.read()
                if len(data) > 0:
                    verified_results.seek(0)
                    self._convert_and_save_results(verified_results)

    def _check_filter_sentences(self, sentence):
        for filter in self._filter_sentences:
            if filter in sentence:
                return True
        return False



