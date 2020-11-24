from DB.schema_definition import AuthorConnection
from commons import commons
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api
from configuration.config_class import getConfig

import logging
from twitter import TwitterError


class Generic_Twitter_Crawler(object):
    def __init__(self, db):
        # AbstractController.__init__(self, db)
        self._db = db
        self._twitter_rest_api = Twitter_Rest_Api(db)
        self._config_parser = getConfig()
        self._domain = str(self._config_parser.get("DEFAULT", "domain"))
        self._users_to_add = []
        self._post_to_add = []

    def retrieve_and_save_data_from_twitter_by_terms(self, keywords, terms, term_desc_term_id_dict, topics):
        posts, total_twitter_users, connections = self.get_posts_and_authors_by_terms(keywords, term_desc_term_id_dict)
        self._db.addPosts(posts)
        self._add_users_to_db(total_twitter_users)

        self._db.addPosts(terms)
        self._db.addPosts(topics)
        self._db.addPosts(connections)


    def commit_db(self):
        self._db.addPosts(self._post_to_add)
        self._add_users_to_db(self._users_to_add)
        self._db.commit()
        self._users_to_add = []
        self._post_to_add = []

    def retrieve_and_save_data_from_twitter_by_post_id(self, post_id, label):
        post, user = self.get_post_and_author_by_post_id(post_id)
        try:
            converted_post = self._db.create_post_from_tweet_data(post, self._domain)
            converted_post.post_type = label
            self._users_to_add.append(user)
            self._post_to_add.append(converted_post)

        except TwitterError as e:
            exception_response = e[0][0]
            logging.info("e.massage =" + exception_response["message"])
            code = exception_response["code"]
            logging.info("e.code =" + str(exception_response["code"]))

            if code == 88:
                sec = self._twitter_rest_api.get_sleep_time_for_twitter_status_id()
                logging.info("Seconds to wait from catched crush is: " + str(sec))
                if sec != 0:
                    commons.count_down_time(sec)
                    self._num_of_twitter_status_id_requests = 0
                return self.retrieve_and_save_data_from_twitter_by_post_id(post_id, label)

    def get_posts_and_authors_by_terms(self, keywords, term_desc_term_id_dict):
        term_tweets_dict = self.get_posts_by_terms(keywords)
        total_twitter_users = []
        total_posts = []
        connections = []
        for term, tweets in term_tweets_dict.items():
            posts = []
            for tweet in tweets:
                post = self._db.create_post_from_tweet_data(tweet, self._domain)
                term_post_connection, term_author_connection = self._create_connections(term, post, term_desc_term_id_dict)
                connections.append(term_post_connection)
                connections.append(term_author_connection)
                posts.append(post)
            total_posts += posts
            #posts += [self._db.create_post_from_tweet_data(tweet, self._domain) for tweet in term_tweets_dict[term]]
            total_twitter_users += [post.user for post in term_tweets_dict[term]]
        return total_posts, total_twitter_users, connections

    def get_post_and_author_by_post_id(self, post_id):
        post = self._twitter_rest_api.get_post_by_post_id(post_id)
        user = post.user
        return post, user

    def _add_users_to_db(self, total_twitter_users):
        author_type = None
        insertion_type = None
        self._twitter_rest_api.save_authors_and_connections(total_twitter_users, author_type, insertion_type)

    def get_posts_by_terms(self, terms):
        return self._twitter_rest_api.get_posts_by_terms(terms)

    def _create_connections(self, term, post, term_desc_term_id_dict):
        term_post_connection = AuthorConnection()

        term_id = term_desc_term_id_dict[term]

        term_post_connection.source_author_guid = term_id
        term_post_connection.destination_author_guid = post.post_id
        term_post_connection.connection_type = "term-post"

        term_author_connection = AuthorConnection()
        term_author_connection.source_author_guid = term_id
        term_author_connection.destination_author_guid = post.author_guid
        term_author_connection.connection_type = "term-author"

        return term_post_connection, term_author_connection


