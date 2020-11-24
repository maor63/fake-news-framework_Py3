
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
from DB.schema_definition import Term, Topic
from commons.method_executor import Method_Executor
from preprocessing_tools.Topic_Term_Manager import Topic_Term_Manager
from .generic_twitter_crawler import Generic_Twitter_Crawler
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api
import itertools

class Twitter_Crawler(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        # taken from http://techslides.com/hacking-the-google-trends-api
        self._url = "https://trends.google.com/trends/hottrends/atom/feed?pn=p1"
        self._retrieve_news_by_keywords = self._config_parser.eval(self.__class__.__name__, "retrieve_news_by_keywords")
        self._retrieve_news_by_terms_and_topics = self._config_parser.eval(self.__class__.__name__, "retrieve_news_by_terms_and_topics")
        self._num_of_top_terms = self._config_parser.eval(self.__class__.__name__, "num_of_top_terms")
        self._generic_twitter_crawler = Generic_Twitter_Crawler(self._db)
        self._topic_term_manager = Topic_Term_Manager(db)
        self._twitter_rest_api = Twitter_Rest_Api(db)

        self._topic_desc_terms_dict = self._retrieve_news_by_terms_and_topics

    def get_most_popular_posts_by_google_trends(self):
        while True:
            terms = self._get_popular_terms_from_google_trends()
            self._generic_twitter_crawler.retrieve_and_save_data_from_twitter_by_terms(terms)

    def get_posts_by_config_keywords(self):
        keywords = self._retrieve_news_by_keywords
        self._create_terms_topics_and_bring_posts(keywords)

    def get_posts_by_terms_and_topics(self):
        terms, keywords, term_desc_term_id_dict = self._convert_keywords_to_terms_by_given_dict()
        topics = self._create_topics_by_given_dict(terms)

        self._generic_twitter_crawler.retrieve_and_save_data_from_twitter_by_terms(keywords, terms, term_desc_term_id_dict, topics)

    def get_posts_by_topic_with_most_posts(self):
        terms = self._topic_term_manager.get_term_from_db_with_most_posts()
        self._generic_twitter_crawler.retrieve_and_save_data_from_twitter_by_terms(terms)

    def _get_popular_terms_from_google_trends(self):
        html_doc = urllib.request.urlopen(self._url)
        soup = BeautifulSoup(html_doc, 'html.parser')
        popular_terms = soup.find_all('title')[1:]
        return [term.text for term in popular_terms]

    def get_posts_by_top_terms_in_each_topic(self):
        'The function returns Twitter posts by top ten terms in each topic. It requires to run Autotopic Executor prior to this'
        # topic = 1, terms = ['isis', 'belgium',...]
        topic_top_terms_dict = self._topic_term_manager.get_topic_top_terms_dictionary(self._num_of_top_terms)

        while True:
            for topic, top_terms in topic_top_terms_dict.items():
                self._generic_twitter_crawler.retrieve_and_save_data_from_twitter_by_terms(top_terms)

    def _convert_keywords_to_terms(self, keywords):
        terms = self._db.get_terms()
        term_term_id_dict = {term.description : term.term_id for term in terms}

        old_terms = set(term_term_id_dict.keys())
        optional_terms = set(keywords)

        keywords_to_add_set = optional_terms - old_terms
        keywords_to_add = list(keywords_to_add_set)

        term_ids = list(term_term_id_dict.values())
        if len(term_ids) > 0:
            max_term_id = max(term_ids)
            new_term_id = max_term_id + 1
        else:
            new_term_id = 1
        terms = []
        for keyword in keywords_to_add:
            term = Term()
            term.term_id = new_term_id

            term_term_id_dict[keyword] = new_term_id

            term.description = keyword
            terms.append(term)
            new_term_id += 1
        return terms, term_term_id_dict

    def _find_term_id_to_assign(self):
        terms = self._db.get_terms()
        term_ids = [term.id for term in terms]
        new_term_id = self._find_id_to_assign(term_ids)
        return new_term_id

    def _find_id_to_assign(self, ids):
        if len(ids) > 0:
            max_id = max(ids)
            new_id = max_id + 1
        else:
            new_id = 1

        return new_id

    def _create_topics(self, terms):
        topics = []
        for term in terms:
            term_id = term.term_id

            topic = Topic()
            topic.topic_id = term_id
            topic.term_id = term_id

            topics.append(topic)
        return topics

    def _create_topics_by_given_dict(self, terms):
        term_desc_term_id_dict = {term.description: term.term_id for term in terms}
        topics = self._db.get_topics()
        topic_ids = [topic.topic_id for topic in topics]

        topic_desc_topic_id_dict = {topic.description: topic.topic_id for topic in topics}

        topic_descriptions = list(topic_desc_topic_id_dict.values())
        optional_topics = list(self._topic_desc_terms_dict.keys())

        topics_to_add = list(set(optional_topics) - set(topic_descriptions))

        topic_index_to_assign = self._find_id_to_assign(topic_ids)

        for topic_to_add in topics_to_add:
            topic_desc_topic_id_dict[topic_to_add] = topic_index_to_assign
            topic_index_to_assign += 1

        new_topics = []
        for topic_description, terms in self._topic_desc_terms_dict.items():
            for term in terms:
                term_id = term_desc_term_id_dict[term]

                topic = Topic()
                topic_id = topic_desc_topic_id_dict[topic_description]
                topic.topic_id = topic_id
                topic.term_id = term_id
                topic.description = topic_description

                new_topics.append(topic)
        return new_topics

    def _create_terms_topics_and_bring_posts(self, keywords):
        terms, term_desc_term_id_dict = self._convert_keywords_to_terms(keywords)
        topics = self._create_topics(terms)

        self._generic_twitter_crawler.retrieve_and_save_data_from_twitter_by_terms(keywords, terms, term_desc_term_id_dict, topics)

    def _convert_keywords_to_terms_by_given_dict(self):
        terms = self._db.get_terms()
        term_desc_term_id_dict = {term.description: term.term_id for term in terms}

        old_terms = set(term_desc_term_id_dict.keys())

        keywords = list(self._topic_desc_terms_dict.values())
        keywords = list(itertools.chain(*keywords))

        optional_terms = set(keywords)

        keywords_to_add_set = optional_terms - old_terms
        keywords_to_add = list(keywords_to_add_set)

        term_ids = list(term_desc_term_id_dict.values())
        if len(term_ids) > 0:
            max_term_id = max(term_ids)
            new_term_id = max_term_id + 1
        else:
            new_term_id = 1
        terms = []
        for keyword in keywords_to_add:
            term = Term()
            term.term_id = new_term_id
            term_desc_term_id_dict[keyword] = new_term_id
            term.description = keyword
            terms.append(term)

            new_term_id += 1
        return terms, keywords, term_desc_term_id_dict
