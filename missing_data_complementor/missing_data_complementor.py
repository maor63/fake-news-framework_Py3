

import re
import timeit
from collections import namedtuple
from collections import namedtuple, Counter
import pandas as pd
from twitter.error import TwitterError

from DB.schema_definition import Post, Author, Post_citation, AuthorConnection
from commons.commons import *
from commons.commons import get_current_time_as_string
from commons.consts import *
from commons.method_executor import Method_Executor
from preprocessing_tools.abstract_controller import AbstractController
from social_network_crawler.social_network_crawler import SocialNetworkCrawler
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api
import csv
import requests

RetweetData = namedtuple('RetweetData', ['retweet_guid', 'retweet_url', 'tweet_guid', 'tweet_url', 'tweet_author_name',
                                         'tweet_author_guid',
                                         'tweet_date', 'tweet_content', 'tweet_twitter_id', 'tweet_retweet_count',
                                         'tweet_favorite_count'])

__author__ = "Aviad Elyashar"


class MissingDataComplementor(Method_Executor):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._actions = self._config_parser.eval(self.__class__.__name__, "actions")

        self._minimal_num_of_posts = self._config_parser.eval(self.__class__.__name__, "minimal_num_of_posts")
        self._limit_friend_follower_number = self._config_parser.eval(self.__class__.__name__,
                                                                      "limit_friend_follower_number")
        self._maximal_tweets_count_in_timeline = self._config_parser.eval(self.__class__.__name__,
                                                                          "maximal_tweets_count_in_timeline")

        self._vico_importer_twitter_authors = []
        self._found_twitter_users = []
        self._social_network_crawler = Twitter_Rest_Api(db)
        self._suspended_authors = []
        self._max_users_without_saving = self._config_parser.eval(self.__class__.__name__, "max_users_without_saving")
        self._max_posts_without_saving = self._config_parser.eval(self.__class__.__name__, "max_users_without_saving")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._account_screen_names = self._config_parser.eval(self.__class__.__name__, "account_screen_names")
        self._posts = []
        self._authors = []
        self._post_citatsions = []


    def setUp(self):
        pass

    def fill_follower_ids(self):
        cursor = self._db.get_followers_or_friends_candidats("follower", self._domain,
                                                             self._limit_friend_follower_number)
        #commented only for 500+ POIs V6 Healthcare provided by Idan Cohen
        followers_or_friends_candidats = self._db.result_iter(cursor)
        followers_or_friends_candidats = [author_id[0] for author_id in followers_or_friends_candidats]


        self._social_network_crawler = SocialNetworkCrawler(self._db)

        self._social_network_crawler.fill_followers_ids_only(followers_or_friends_candidats)


    def fill_followers_ids_for_POIs(self):

        authors_ids = self._db.get_author_ids_not_general_public_and_not_brought_followers_for_them()

        self._social_network_crawler = SocialNetworkCrawler(self._db)

        self._social_network_crawler.fill_followers_ids_only(authors_ids)


    def convert_temp_author_connections_into_author_connections(self):
        self._db.convert_temp_author_connections_to_author_connections(self._domain)


    def fill_followers_and_their_data_simultaneously(self):
        cursor = self._db.get_followers_or_friends_candidats("follower", self._domain,
                                                             self._limit_friend_follower_number)

        followers_or_friends_candidats = self._db.result_iter(cursor)
        followers_or_friends_candidats = [author_id[0] for author_id in followers_or_friends_candidats]

        self._social_network_crawler = SocialNetworkCrawler(self._db)

        self._social_network_crawler.fill_followers_and_their_data_simultaneously(followers_or_friends_candidats)


    def fill_data_for_followers(self):
        self._fill_data_for_author_connection_type(Author_Connection_Type.FOLLOWER)
        logging.info("---Finished crawl_followers_by_author_ids")

    def fill_followers_for_authors(self):
        cursor = self._db.get_followers_or_friends_candidats("follower", self._domain,
                                                             self._limit_friend_follower_number)
        followers_or_friends_candidats = self._db.result_iter(cursor)
        followers_or_friends_candidats = [author_id[0] for author_id in followers_or_friends_candidats]

        # followers_or_friends_candidats = ['19583545']

        print("---crawl_followers_by_author_ids---")
        author_type = None
        are_user_ids = True
        insertion_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        # crawl_users_by_author_ids_func_name = "crawl_users_by_author_ids"
        self._social_network_crawler = SocialNetworkCrawler(self._db)
        self._social_network_crawler.crawl_users_by_author_ids(followers_or_friends_candidats,
                                                               "follower", author_type,
                                                               are_user_ids, insertion_type)

        self._db.convert_temp_author_connections_to_author_connections(self._domain)

    #######
    # In this case we will collect followers until we got an exception.
    # When exception is raised we start asking for users until exception or finish.
    # Save in the DB. Remoove
    #######

    def fill_followers_for_authors_handle_exceptions(self):
        # cursor = self._db.get_followers_or_friends_candidats("follower", self._domain,
        #                                                      self._limit_friend_follower_number)
        # followers_or_friends_candidats = self._db.result_iter(cursor)
        # followers_or_friends_candidats = [author_id[0] for author_id in followers_or_friends_candidats]

        user_ids_to_bring_followers = self._db.get_followers_brought_by_terms()

        print("---crawl_followers_by_author_ids---")
        author_type = None
        are_user_ids = True
        insertion_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        # crawl_users_by_author_ids_func_name = "crawl_users_by_author_ids"
        self._social_network_crawler = SocialNetworkCrawler(self._db)

        total_already_checked_author_ids = []
        candidates = user_ids_to_bring_followers
        while len(user_ids_to_bring_followers) > len(total_already_checked_author_ids):
            print("{0} > {1}".format(len(user_ids_to_bring_followers), len(total_already_checked_author_ids)))
            tweeter_user_ids, already_checked_author_ids = self._social_network_crawler.crawl_users_by_author_ids(
                candidates,
                "follower", author_type,
                are_user_ids, insertion_type)

            total_already_checked_author_ids += already_checked_author_ids
            candidates = list(set(user_ids_to_bring_followers) - set(total_already_checked_author_ids))

        self._db.convert_temp_author_connections_to_author_connections(self._domain)

    def fill_data_for_friends(self):
        self._fill_data_for_author_connection_type(Author_Connection_Type.FRIEND)
        logging.info("---Finished crawl_friends_by_author_ids")

    def _fill_data_for_author_connection_type(self, connection_type):
        cursor = self._db.get_followers_or_friends_candidats(connection_type, self._domain,
                                                             self._limit_friend_follower_number)
        followers_or_friends_candidats = self._db.result_iter(cursor)
        followers_or_friends_candidats = [author_id[0] for author_id in followers_or_friends_candidats]
        print("---crawl_followers_by_author_ids---")
        author_type = None
        are_user_ids = True
        insertion_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        crawl_users_by_author_ids_func_name = "crawl_users_by_author_ids"
        getattr(self._social_network_crawler, crawl_users_by_author_ids_func_name)(followers_or_friends_candidats,
                                                                                   connection_type, author_type,
                                                                                   are_user_ids, insertion_type)
        self._db.convert_temp_author_connections_to_author_connections(self._domain)

    def crawl_followers_by_author_ids(self, author_ids):
        print("---crawl_followers_by_author_ids---")
        author_type = None
        are_user_ids = True
        inseration_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        self._social_network_crawler.crawl_followers_by_twitter_author_ids(author_ids, author_type, are_user_ids,
                                                                           inseration_type)

    def crawl_friends_by_author_ids(self, author_ids):
        print("---crawl_friends_by_author_ids---")
        author_type = None
        are_user_ids = True
        inseration_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        self._social_network_crawler.crawl_friends_by_twitter_author_ids(author_ids, author_type, are_user_ids,
                                                                         inseration_type)

    def get_twitter_authors_retrieved_from_vico_importer(self):
        authors = self._db.get_twitter_authors_retrieved_from_vico_importer()
        self._vico_importer_twitter_authors = authors
        return authors

    def create_author_screen_names(self):
        screen_names = self._db.get_screen_names_for_twitter_authors_by_posts()
        return screen_names

    def fill_data_for_sources(self):
        print("---complete_missing_information_for_authors_by_screen_names ---")
        #twitter_author_screen_names = self._db.get_missing_data_twitter_screen_names_by_posts()
        #twitter_author_screen_names = self._db.get_hospital_twitter_screen_names()
        #twitter_author_screen_names = self._db.get_labor_employees_screen_names()
        #twitter_author_screen_names = self._db.get_follower_ids_of_hospitals_and_labor_unions()
        #twitter_author_screen_names = self._db.get_intersection_of_labor_union_and_healthcare_users_followers_ids()
        #twitter_author_screen_names = self._db.get_healthcare_labor_union_follower_ids()
        #twitter_author_screen_names = self._db.get_poi_screen_names()
        twitter_author_screen_names = self._db.get_follower_ids_to_crawl()
        #twitter_author_screen_names = [int(i) for i in twitter_author_screen_names]
        # brought about 250+ POIs from Idan Cohen
        #twitter_author_screen_names = self._db.get_poi_v6_screen_names()


        self._fill_data_for_authors_by_screen_names(twitter_author_screen_names)
        #self.fill_author_guid_to_posts()

    # before run it please add the csv file for spokenmanships as a table.

    def fill_data_for_spokesmanships(self):
        print("fill_data_for_spokesmanships")

        spokesmanships_screen_names = self._db.get_spokesmanships_screen_names()
        self._fill_data_for_authors_by_screen_names(spokesmanships_screen_names)


    def fill_data_for_authors(self):
        print("---complete_missing_information_for_authors_by_screen_names ---")
        twitter_author_screen_names = self._db.get_missing_data_twitter_screen_names_by_authors()

        self._fill_data_for_authors_by_screen_names(twitter_author_screen_names)

    def _fill_data_for_authors_by_screen_names(self, twitter_author_screen_names):
        logging.info("---complete_missing_information_for_authors_by_screen_names ---")
        # twitter_author_screen_names = self.create_author_screen_names()
        # twitter_author_screen_names = self._db.get_missing_data_twitter_screen_names()
        # twitter_author_screen_names = (twitter_author.name for twitter_author in twitter_authors)
        # twitter_author_screen_names = list(twitter_author_screen_names)
        author_type = None
        are_user_ids = False
        inseration_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        # retrieve_full_data_for_missing_users
        i = 1
        for author_screen_names in split_into_equal_chunks(twitter_author_screen_names, self._max_users_without_saving):
            twitter_users = self._social_network_crawler.handle_get_users_request(
                author_screen_names, are_user_ids, author_type, inseration_type)

            print('retrieve authors {0}/{1} total:{2}'.format(i * self._max_users_without_saving,
                                                  len(twitter_author_screen_names), len(twitter_author_screen_names)))
            i += 1
            self._social_network_crawler.save_authors_and_connections(twitter_users, author_type, inseration_type)
        # self.fill_author_guid_to_posts()

        # self._db.delete_posts_with_missing_authors()
        # self.insert_suspended_accounts()
        # self.label_verified_accounts_as_good_actors()
        print("---complete_missing_information_for_authors_by_screen_names was completed!!!!---")
        logging.info("---complete_missing_information_for_authors_by_screen_names was completed!!!!---")
        # return total_twitter_users

    def fill_data_for_temp_authors(self):
        temp_author_connections = self._db.get_temp_author_connections_all()
        author_ids = [temp_author_connection.destination_author_osn_id for temp_author_connection in
                      temp_author_connections]

        author_type = None
        are_user_ids = True
        inseration_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        # retrieve_full_data_for_missing_users
        i = 1
        for author_ids in split_into_equal_chunks(author_ids, self._max_users_without_saving):
            twitter_users = self._social_network_crawler.handle_get_users_request(
                author_ids, are_user_ids, author_type, inseration_type)

            print('retrieve authors {}/{}'.format(i * self._max_users_without_saving,
                                                  len(author_ids)))
            i += 1
            self._social_network_crawler.save_authors_and_connections(twitter_users, author_type, inseration_type)

        all_authors = self._db.get_authors()
        author_osn_id_author_guid_dict = {author.author_osn_id: author.author_guid for author in all_authors}

        author_connections = []
        for temp_author_connection in temp_author_connections:
            source_author_osn_id = temp_author_connection.source_author_osn_id
            destination_author_osn_id = temp_author_connection.destination_author_osn_id

            if source_author_osn_id in author_osn_id_author_guid_dict and destination_author_osn_id in author_osn_id_author_guid_dict:
                source_author_guid = author_osn_id_author_guid_dict[source_author_osn_id]
                destination_author_guid = author_osn_id_author_guid_dict[destination_author_osn_id]

                author_connection = AuthorConnection()

                author_connection.source_author_guid = source_author_guid
                author_connection.destination_author_guid = destination_author_guid
                author_connection.connection_type = "follower"
                author_connections.append(author_connection)

        self._db.addPosts(author_connections)
        self._db.delete_temp_author_connections(temp_author_connections)

    def fill_author_guid_to_posts(self):
        posts = self._db.get_posts()
        num_of_posts = len(posts)
        for i, post in enumerate(posts):
            msg = "\rPosts to fill: [{0}/{1}]".format(i, num_of_posts)
            print(msg, end="")
            post.author_guid = compute_author_guid_by_author_name(post.author)
        self._db.addPosts(posts)
        self._db.insert_or_update_authors_from_posts(self._domain, {}, {})

    def complete_missing_information_for_authors_by_ids(self):
        print("---complete_missing_information_for_authors_by_ids ---")
        logging.info("---complete_missing_information_for_authors_by_ids ---")
        # twitter_author_screen_names = self.create_author_screen_names()
        twitter_author_screen_names = self._db.get_missing_data_twitter_screen_names()
        # twitter_author_screen_names = (twitter_author.name for twitter_author in twitter_authors)
        # twitter_author_screen_names = list(twitter_author_screen_names)

        author_type = None
        are_user_ids = False
        inseration_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
        # retrieve_full_data_for_missing_users
        total_twitter_users = self._social_network_crawler.handle_get_users_request(
            twitter_author_screen_names, are_user_ids, author_type, inseration_type)
        # return self._found_twitter_users
        print("---complete_missing_information_for_authors was completed!!!!---")
        logging.info("---complete_missing_information_for_authors was completed!!!!---")
        return total_twitter_users

    def mark_suspended_or_not_existed_authors(self):
        suspended_authors = self._db.get_authors_for_mark_as_suspended_or_not_existed()
        for suspended_author in suspended_authors:
            suspended_author.is_suspended_or_not_exists = self._window_start
            self._db.set_inseration_date(suspended_author, DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR)
        self._social_network_crawler.save_authors(suspended_authors)

    def mark_suspended_from_twitter(self):
        self._suspended_authors = []
        suspected_authors = self._db.get_not_suspended_authors(self._domain)
        suspected_authors_names = [author.name for author in suspected_authors]
        chunks = split_into_equal_chunks(suspected_authors_names,
                                         self._social_network_crawler._maximal_user_ids_allowed_in_single_get_user_request)
        total_chunks = list(chunks)
        chunks = split_into_equal_chunks(suspected_authors_names,
                                         self._social_network_crawler._maximal_user_ids_allowed_in_single_get_user_request)
        i = 1
        for chunk_of_names in chunks:
            msg = "\rChunck of author to Twitter: [{0}/{1}]".format(i, len(total_chunks))
            print(msg, end="")
            i += 1
            set_of_send_author_names = set(chunk_of_names)
            set_of_received_author_names = set(
                self._social_network_crawler.get_active_users_names_by_screen_names(chunk_of_names))
            author_names_of_suspendend_or_not_exists = set_of_send_author_names - set_of_received_author_names
            self._update_suspended_authors_by_screen_names(author_names_of_suspendend_or_not_exists)
        self._db.add_authors(self._suspended_authors)

    def _update_suspended_authors_by_screen_names(self, author_names_of_suspendend_or_not_exists):
        for author_name in author_names_of_suspendend_or_not_exists:
            user_guid = compute_author_guid_by_author_name(author_name).replace("-", "")
            suspended_author = self._db.get_author_by_author_guid(user_guid)

            suspended_author.is_suspended_or_not_exists = self._window_start
            suspended_author.author_type = Author_Type.BAD_ACTOR
            self._db.set_inseration_date(suspended_author, DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR)
            self._suspended_authors.append(suspended_author)

            num_of_suspended_authors = len(self._suspended_authors)
            if num_of_suspended_authors == self._max_users_without_saving:
                self._db.add_authors(self._suspended_authors)
                self._suspended_authors = []

    def fill_tweet_retweet_connection(self):
        '''
        Fetches the original tweets being retweeted by our posts.
        Updates the followig tables:
         * Post_Citations table with tweet-retweet connection
         * Posts table with missing tweets
         * Authors with the authors of the missing tweets
        '''
        retweets_with_no_tweet_citation = self._db.get_retweets_with_no_tweet_citation()
        logging.info("Updating tweet-retweet connection of {0} retweets".format(len(retweets_with_no_tweet_citation)))
        self._posts = []
        self._authors = []
        self._post_citatsions = []
        i = 1
        for post_guid, post_url in retweets_with_no_tweet_citation.items():
            # logging.info("Analyzing retweet: {0} - {1}".format(post_guid, post_url))
            msg = "\r Analyzing retweet: {0} - {1} [{2}".format(post_guid, post_url, i) + "/" + str(
                len(retweets_with_no_tweet_citation)) + '] '
            print(msg, end="")
            i += 1
            tweet_data = self.extract_retweet_data(retweet_guid=post_guid, retweet_url=post_url)
            if tweet_data is not None:

                if not self._db.isPostExist(tweet_data.tweet_url):
                    post = Post(guid=tweet_data.tweet_guid, post_id=tweet_data.tweet_guid, url=tweet_data.tweet_url,
                                date=str_to_date(tweet_data.tweet_date),
                                title=tweet_data.tweet_content, content=tweet_data.tweet_content,
                                post_osn_id=tweet_data.tweet_twitter_id,
                                retweet_count=tweet_data.tweet_retweet_count,
                                favorite_count=tweet_data.tweet_favorite_count,
                                author=tweet_data.tweet_author_name, author_guid=tweet_data.tweet_author_guid,
                                domain=self._domain,
                                original_tweet_importer_insertion_date=str(get_current_time_as_string()))
                    self._posts.append(post)

                if not self._db.is_author_exists(tweet_data.tweet_author_guid, self._domain):
                    author = Author(name=tweet_data.tweet_author_name,
                                    domain=self._domain,
                                    author_guid=tweet_data.tweet_author_guid,
                                    original_tweet_importer_insertion_date=str(get_current_time_as_string()))
                    self._authors.append(author)

                if not self._db.is_post_citation_exist(tweet_data.retweet_guid, tweet_data.tweet_guid):
                    post_citation = Post_citation(post_id_from=tweet_data.retweet_guid,
                                                  post_id_to=tweet_data.tweet_guid, url_from=tweet_data.retweet_url,
                                                  url_to=tweet_data.tweet_url)
                    self._post_citatsions.append(post_citation)

        self.update_tables_with_tweet_retweet_data(self._posts, self._authors, self._post_citatsions)

    def extract_retweet_data(self, retweet_guid, retweet_url):
        '''
        :param retweet_guid: the guid of the retweet
        :param retweet_url: the url of the retweet
        :return: a RetweetData holding the data of the retweet
        '''
        try:
            retweet_id = self.extract_tweet_id(retweet_url)
            if retweet_id is None:
                return None

            retweet_status = self._social_network_crawler.get_status_by_twitter_status_id(retweet_id)
            tweet_status_dict = retweet_status.AsDict()
            if 'retweeted_status' in tweet_status_dict:
                tweet_status_dict = tweet_status_dict['retweeted_status']
                tweet_post_twitter_id = str(str(tweet_status_dict['id']))
                tweet_author_name = str(tweet_status_dict['user']['screen_name'])
                tweet_url = str(generate_tweet_url(tweet_post_twitter_id, tweet_author_name))
                tweet_creation_time = str(tweet_status_dict['created_at'])
                tweet_str_publication_date = str(extract_tweet_publiction_date(tweet_creation_time))
                tweet_guid = str(compute_post_guid(post_url=tweet_url, author_name=tweet_author_name,
                                                       str_publication_date=tweet_str_publication_date))
                tweet_author_guid = str(compute_author_guid_by_author_name(tweet_author_name))
                tweet_author_guid = str(tweet_author_guid.replace("-", ""))
                tweet_content = str(tweet_status_dict['text'])
                tweet_retweet_count = str(tweet_status_dict['retweet_count'])
                tweet_favorite_count = str(tweet_status_dict['favorite_count'])

                retweet_data = RetweetData(retweet_guid=retweet_guid, retweet_url=retweet_url, tweet_guid=tweet_guid,
                                           tweet_url=tweet_url, tweet_author_name=tweet_author_name,
                                           tweet_author_guid=tweet_author_guid,
                                           tweet_date=tweet_str_publication_date, tweet_content=tweet_content,
                                           tweet_twitter_id=tweet_post_twitter_id,
                                           tweet_retweet_count=tweet_retweet_count,
                                           tweet_favorite_count=tweet_favorite_count)
                return retweet_data
            else:
                return None

        except TwitterError as e:
            exception_response = e[0][0]
            logging.info("e.massage =" + exception_response["message"])
            code = exception_response["code"]
            logging.info("e.code =" + str(exception_response["code"]))

            self.update_tables_with_tweet_retweet_data(self._posts, self._authors, self._post_citatsions)
            self._posts = []
            self._authors = []
            self._post_citatsions = []

            if code == 88:
                sec = self._social_network_crawler.get_sleep_time_for_twitter_status_id()
                logging.info("Seconds to wait from catched crush is: " + str(sec))
                if sec != 0:
                    count_down_time(sec)
                    self._num_of_twitter_status_id_requests = 0
                return self._social_network_crawler.get_status(retweet_id)

        except Exception as e:
            logging.error("Cannot fetch data for retweet: {0}. Error message: {1}".format(retweet_url, e.message))
            return None

    def extract_tweet_id(self, post_url):
        pattern = re.compile("http(.*)://twitter.com/(.*)/statuses/(.*)")
        extracted_info = pattern.findall(post_url)
        if extracted_info == []:
            pattern = re.compile("http(.*)://twitter.com/(.*)/status/(.*)")
            extracted_info = pattern.findall(post_url)
        if len(extracted_info[0]) < 2:
            return None
        else:
            return extracted_info[0][2]

    def update_tables_with_tweet_retweet_data(self, posts, authors, post_citatsions):
        self._db.addPosts(posts)
        self._db.add_authors(authors)
        self._db.addReferences(post_citatsions)

    def fill_authors_time_line(self):
        '''
        Fetches the posts for the authors that are given under authors_twitter_ids_for_timeline_filling in the config file +
        update the db
        '''
        self._db.create_authors_index()
        self._db.create_posts_index()
        author_screen_name_dict = self._db.authors_dict_by_field('author_screen_name')
        early_stop = False
        for i in range(100):
            author_screen_names_number_of_posts = self._db.get_author_screen_names_and_number_of_posts(
                self._minimal_num_of_posts)

            if not author_screen_names_number_of_posts:
                early_stop = True

            author_screen_names_number_of_posts_dict = self._create_author_screen_name_number_of_posts_dictionary(
                author_screen_names_number_of_posts)

            if not author_screen_names_number_of_posts_dict:
                author_screen_names_number_of_posts_dict = {}
                for author in self._db.get_authors():
                    author_screen_names_number_of_posts_dict[author.author_screen_name] = 0

            if early_stop:
                author_screen_name_last_post_id = self._db.get_author_screen_name_first_post_id()
            else:
                author_screen_name_last_post_id = self._db.get_author_screen_name_last_post_id()
            index = 1
            for author_name in author_screen_names_number_of_posts_dict:
                print("Get timeline for {0} : {1}/{2}".format(author_name, str(index),
                                                              str(len(author_screen_names_number_of_posts_dict))))
                index += 1
                posts = []
                logging.info("Fetching timeline for author: " + str(author_name))
                posts_counter = 0
                try:
                    posts_needed_from_osn = self._minimal_num_of_posts - author_screen_names_number_of_posts_dict[
                        author_name]
                    if author_name in author_screen_name_last_post_id:
                        last_post_id = author_screen_name_last_post_id[author_name]
                        if early_stop:
                            timeline = self._social_network_crawler.get_timeline_by_author_name(author_name,
                                                                                                posts_needed_from_osn,
                                                                                                since_id=last_post_id)
                        else:
                            timeline = self._social_network_crawler.get_timeline_by_author_name(author_name,
                                                                                                posts_needed_from_osn,
                                                                                                max_id=last_post_id)
                    else:
                        timeline = self._social_network_crawler.get_timeline_by_author_name(author_name,
                                                                                            posts_needed_from_osn)
                    # logging.info("Retrived timeline lenght: " + str(len(timeline)))
                    if timeline is not None:
                        for tweet in timeline:
                            # post = self._convert_tweet_to_post(tweet)
                            # post = self._convert_tweet_to_post(tweet)
                            post = self._db.create_post_from_tweet_data(tweet, self._domain)
                            if len(timeline) == 1:
                                author = author_screen_name_dict[author_name]
                                author.timeline_overlap_insertion_date = post.date
                                posts.append(author)
                            posts.append(post)
                except Exception as e:
                    logging.error(
                        "Cannot fetch data for author: {0}. Error message: {1}".format(author_name, e.message))
                # logging.info("Number of posts inserted for author {0}: {1}".format(author_name, posts_counter))
                self._db.addPosts(posts)
            if early_stop:
                break

    def assign_manually_labeled_authors(self):
        self._db.assign_manually_labeled_authors()

    def delete_acquired_authors(self):
        self._db.delete_acquired_authors()
        self._db.delete_posts_with_missing_authors()

    def delete_manually_labeled_authors(self):
        self._db.delete_manually_labeled_authors()
        self._db.delete_posts_with_missing_authors()

    def assign_acquired_and_crowd_turfer_profiles(self):
        self._db.assign_crowdturfer_profiles()
        self._db.assign_acquired_profiles()

    def _create_author_screen_name_number_of_posts_dictionary(self, author_screen_names_number_of_posts):
        author_screen_names_number_of_posts_dict = {}
        for record in author_screen_names_number_of_posts:
            author_screen_name = record[0]
            num_of_posts = record[1]
            author_screen_names_number_of_posts_dict[author_screen_name] = num_of_posts
        logging.info("Number of users to retrieve timelines: " + str(len(author_screen_names_number_of_posts_dict)))
        return author_screen_names_number_of_posts_dict

    ###
    # There is an option that there are posts that twitter did not provide their information.
    # Most of them are suspended or protected.
    # In this case we will add them into the authors table with label of 'bad_actor'.
    # We should fill the author_guid in the posts table.
    ###
    def insert_suspended_accounts(self):
        authors = []
        author_screen_names = []
        author_guids = []
        missing_author_posts = self._db.get_posts_of_missing_authors()
        num_of_missing_posts = len(missing_author_posts)
        for i, missing_author_post in enumerate(missing_author_posts):
            msg = "\rInserting missing authors to authors table: {0}/{1}".format(i, num_of_missing_posts)
            print(msg, end="")

            author = Author()

            author_screen_name = missing_author_post.author
            author.author_screen_name = author_screen_name
            author.name = author_screen_name
            author_screen_names.append(author_screen_name)

            author_guid = compute_author_guid_by_author_name(author_screen_name)
            author.author_guid = author_guid
            author_guids.append(author_guid)

            author.author_type = "bad_actor"

            author.domain = self._domain

            authors.append(author)

            # update the missing guid to post
            missing_author_post.author_guid = author_guid

        self._db.add_authors(authors)
        self._db.addPosts(missing_author_posts)

        with open(self._output_path + "insert_suspended_accounts.txt", 'w') as output_file:
            output_file.write("Number of suspended_users_added_to_authors_table is: " + str(num_of_missing_posts))
            output_file.write("\n")

            author_screen_names_str = ','.join(author_screen_names)
            output_file.write("author_screen_names: " + author_screen_names_str)
            output_file.write("\n")

            author_guids_str = ','.join(author_guids)
            output_file.write("author_guids: " + author_guids_str)

    #
    # This function was created for filling information in Arabic Honeypot dataset
    #
    def insert_suspended_accounts2(self):
        authors = []
        author_screen_names = []
        author_guids = []
        author_guid_author_screen_name_tuples = self._db.get_missing_authors_tuples()
        author_guid_author_screen_name_tuples = list(author_guid_author_screen_name_tuples)
        num_of_suspended_accounts = len(author_guid_author_screen_name_tuples)
        for i, author_guid_author_screen_name_tuple in enumerate(author_guid_author_screen_name_tuples):
            msg = "\rInserting missing authors to authors table: {0}/{1}".format(i, num_of_suspended_accounts)
            print(msg, end="")

            author_guid = author_guid_author_screen_name_tuple[0]
            author_screen_name = author_guid_author_screen_name_tuple[1]
            if author_guid is None and author_screen_name is None:
                continue

            author = Author()

            author.author_screen_name = author_screen_name
            author.name = author_screen_name
            author_screen_names.append(author_screen_name)

            if author_guid is None:
                author_guid = compute_author_guid_by_author_name(author_screen_name)
            author.author_guid = author_guid
            author_guids.append(author_guid)

            author.author_type = "bad_actor"

            author.domain = self._domain

            authors.append(author)

        self._db.add_authors(authors)

        with open(self._output_path + "insert_suspended_accounts.csv", 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow("Number of suspended_users_added_to_authors_table is: " + str(num_of_suspended_accounts))

            author_screen_names_str = ','.join(author_screen_names)
            writer.writerow("author_screen_names: " + author_screen_names_str)

            author_guids_str = ','.join(author_guids)
            writer.writerow("author_guids: " + author_guids_str)

    def label_verified_accounts_as_good_actors(self):
        verified_authors = self._db.get_verified_authors()
        num_of_verified_authors = len(verified_authors)
        for i, verified_author in enumerate(verified_authors):
            msg = "\rLabel verified accounts as good actors: {0}/{1}".format(i, num_of_verified_authors)
            print(msg, end="")

            verified_author.author_type = "good_actor"

        self._db.add_authors(verified_authors)

        with open(self._output_path + "label_verified_accounts_as_good_actors.txt", 'w') as output_file:
            output_file.write("Number of verified authors is: " + str(num_of_verified_authors))
            output_file.write("\n")

    def fill_retweets_for_tweets(self):
        posts = self._db.get_posts()
        num_of_posts = len(posts)
        posts_to_add = []
        authors_to_add = []
        user_mentions_to_add = []
        for i, post in enumerate(posts):
            msg = "\rGiven posts for filling retweets : [{0}/{1}]".format(i, num_of_posts)
            print(msg, end="")

            tweet_id = post.post_osn_id
            if tweet_id is not None:
                retweets = self._social_network_crawler.get_retweets_by_post_id(tweet_id)

                for retweet in retweets:
                    post, author = self._db._convert_tweet_to_post_and_author(retweet, self._domain)
                    user_mentions = self._db.convert_tweet_to_user_mentions(retweet, post.guid)
                    user_mentions_to_add += user_mentions
                    posts_to_add.append(post)
                    authors_to_add.append(author)

        self._db.addPosts(posts_to_add)
        self._db.addPosts(authors_to_add)
        self._db.addPosts(user_mentions_to_add)

    def collect_timeline_posts_for_author_screen_names(self):
        self._convert_timeline_tweets_to_posts_for_author_screen_names(self._account_screen_names)

    def _convert_timeline_tweets_to_posts_for_author_screen_names(self, author_screen_names):
        posts = []
        for i, account_screen_name in enumerate(author_screen_names):
            try:
                # timeline_tweets = self._social_network_crawler.get_timeline_by_author_name(account_screen_name, 3200)
                timeline_tweets = self._social_network_crawler.get_timeline(account_screen_name, 3200)
                if timeline_tweets is not None:
                    print("\rSearching timeline tweets for author_guid: {0} {1}/{2} retrieved:{3}".format(
                        account_screen_name, i,
                        len(author_screen_names), len(timeline_tweets)),
                        end='')
                    for timeline_tweet in timeline_tweets:
                        post = self._db.create_post_from_tweet_data(timeline_tweet, self._domain)
                        posts.append(post)

                    if len(posts) > self._max_posts_without_saving:
                        self._db.addPosts(posts)
                        posts = []
            except requests.exceptions.ConnectionError as errc:
                x = 3
                # print("Error Connecting:", errc)
                # print("Reconnecting ...")
                # self._social_network_crawler = SocialNetworkCrawler(self._db)
                #
                # author_connections = self._collect_followers_and_save_connections(user_id,
                #                                                                   author_osn_id_author_guid_dict)

            except TwitterError as e:
                if e.message == "Not authorized.":
                    logging.info("Not authorized for user id: {0}".format(account_screen_name))
                    continue

        self._db.addPosts(posts)

    def collect_timeline_for_all_authors(self):
        authors = self._db.get_authors()
        author_screen_names = [author.author_screen_name for author in authors]

        author_screen_names_for_timeline = set(author_screen_names) - set(self._account_screen_names)
        author_screen_names_for_timeline = list(author_screen_names_for_timeline)
        self._convert_timeline_tweets_to_posts_for_author_screen_names(author_screen_names_for_timeline)

    def collect_timeline_for_healthcare_workers_assigned_automatically(self):
        author_screen_names_for_timeline = self._db.get_healthcare_worker_screen_names()
        self._convert_timeline_tweets_to_posts_for_author_screen_names(author_screen_names_for_timeline)

    def fill_follower_connections_among_authors_only_for_commnets_authors(self):
        self._connection_type = "follower"

        authors = self._db.get_authors()
        author_osn_id_author_guid_dict = {author.author_osn_id: author.author_guid for author in authors}
        author_osn_id_author_dict = {author.author_osn_id: author for author in authors if
                                     author.protected == 0}

        # cursor = self._db.get_followers_or_friends_candidats("follower", self._domain,
        #                                                      self._limit_friend_follower_number)
        # user_ids_to_fill_followers = self._db.result_iter(cursor)
        # user_ids_to_fill_followers = [author_id[0] for author_id in user_ids_to_fill_followers]
        user_ids_to_fill_followers = self._db.get_authors_guid_by_comments()

        self._social_network_crawler = SocialNetworkCrawler(self._db)

        total_author_connections = []
        finished_with_authors = []
        requests_count = 0
        # user_ids_to_fill_followers = [user_ids_to_fill_followers[0]]
        for i, user_id in enumerate(user_ids_to_fill_followers):
            print("\rSearching followers for author_guid: {0} {1}/{2}".format(user_id, i,
                                                                              len(user_ids_to_fill_followers)), end='')

            author = author_osn_id_author_dict[user_id]
            insertion_date = str(get_current_time_as_string())
            author.vico_dump_insertion_date = insertion_date
            finished_with_authors.append(author)

            if requests_count < 14:
                author_connections = self._collect_follower_ids_for_user_and_save_connections(user_id,
                                                                                              author_osn_id_author_guid_dict)
                total_author_connections += author_connections

            else:
                requests_count = 0
                self._db.addPosts(total_author_connections)
                self._db.addPosts(finished_with_authors)
                total_author_connections = []
                finished_with_authors = []
                count_down_time(15 * 60)

                author_connections = self._collect_follower_ids_for_user_and_save_connections(user_id,
                                                                                              author_osn_id_author_guid_dict)
                total_author_connections += author_connections

            requests_count += 1

        self._db.addPosts(total_author_connections)
        self._db.addPosts(finished_with_authors)

    def _collect_follower_ids_for_user_and_save_connections(self, user_id, author_osn_id_author_guid_dict):
        try:
            author_connections = []
            author_connections = self._collect_followers_and_save_connections(user_id, author_osn_id_author_guid_dict)

        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            print("Reconnecting ...")
            self._social_network_crawler = SocialNetworkCrawler(self._db)

            author_connections = self._collect_followers_and_save_connections(user_id, author_osn_id_author_guid_dict)

        except TwitterError as e:
            if e.message == "Not authorized.":
                logging.info("Not authorized for user id: {0}".format(user_id))

        return author_connections

    def fill_follower_connections_among_authors_only_for_commnets_authors(self):
        self._connection_type = "follower"

        authors = self._db.get_authors()
        author_osn_id_author_guid_dict = {author.author_osn_id: author.author_guid for author in authors}
        author_osn_id_author_dict = {author.author_osn_id: author for author in authors if
                                     author.protected == 0}

        # cursor = self._db.get_followers_or_friends_candidats("follower", self._domain,
        #                                                      self._limit_friend_follower_number)
        # user_ids_to_fill_followers = self._db.result_iter(cursor)
        # user_ids_to_fill_followers = [author_id[0] for author_id in user_ids_to_fill_followers]
        user_ids_to_fill_followers = self._db.get_authors_guid_by_comments()

        self._social_network_crawler = SocialNetworkCrawler(self._db)

        total_author_connections = []
        finished_with_authors = []
        requests_count = 0
        # user_ids_to_fill_followers = [user_ids_to_fill_followers[0]]
        for i, user_id in enumerate(user_ids_to_fill_followers):
            print("\rSearching followers for author_guid: {0} {1}/{2}".format(user_id, i,
                                                                              len(user_ids_to_fill_followers)), end='')

            author = author_osn_id_author_dict[user_id]
            insertion_date = str(get_current_time_as_string())
            author.vico_dump_insertion_date = insertion_date
            finished_with_authors.append(author)

            if requests_count < 14:
                author_connections = self._collect_follower_ids_for_user_and_save_connections(user_id,
                                                                                              author_osn_id_author_guid_dict)
                total_author_connections += author_connections

            else:
                requests_count = 0
                self._db.addPosts(total_author_connections)
                self._db.addPosts(finished_with_authors)
                total_author_connections = []
                finished_with_authors = []
                count_down_time(15 * 60)

                author_connections = self._collect_follower_ids_for_user_and_save_connections(user_id,
                                                                                              author_osn_id_author_guid_dict)
                total_author_connections += author_connections

            requests_count += 1

        self._db.addPosts(total_author_connections)
        self._db.addPosts(finished_with_authors)

    def fill_follower_connections_among_authors(self):
        self._connection_type = "follower"

        authors = self._db.get_authors()
        author_osn_id_author_guid_dict = {author.author_osn_id: author.author_guid for author in authors}
        author_osn_id_author_dict = {author.author_osn_id: author for author in authors if
                                     author.vico_dump_insertion_date is None
                                     and author.protected == 0}

        # cursor = self._db.get_followers_or_friends_candidats("follower", self._domain,
        #                                                      self._limit_friend_follower_number)
        # user_ids_to_fill_followers = self._db.result_iter(cursor)
        # user_ids_to_fill_followers = [author_id[0] for author_id in user_ids_to_fill_followers]
        user_ids_to_fill_followers = list(author_osn_id_author_dict.keys())

        self._social_network_crawler = SocialNetworkCrawler(self._db)

        total_author_connections = []
        finished_with_authors = []
        requests_count = 0
        # user_ids_to_fill_followers = [user_ids_to_fill_followers[0]]
        for i, user_id in enumerate(user_ids_to_fill_followers):
            print("\rSearching followers for author_guid: {0} {1}/{2}".format(user_id, i,
                                                                              len(user_ids_to_fill_followers)), end='')

            author = author_osn_id_author_dict[user_id]
            insertion_date = str(get_current_time_as_string())
            author.vico_dump_insertion_date = insertion_date
            finished_with_authors.append(author)

            if requests_count < 14:
                author_connections = self._collect_follower_ids_for_user_and_save_connections(user_id,
                                                                                              author_osn_id_author_guid_dict)
                total_author_connections += author_connections

            else:
                requests_count = 0
                self._db.addPosts(total_author_connections)
                self._db.addPosts(finished_with_authors)
                total_author_connections = []
                finished_with_authors = []
                count_down_time(15 * 60)

                author_connections = self._collect_follower_ids_for_user_and_save_connections(user_id,
                                                                                              author_osn_id_author_guid_dict)
                total_author_connections += author_connections

            requests_count += 1

        self._db.addPosts(total_author_connections)
        self._db.addPosts(finished_with_authors)

    def fill_retweets_for_posts(self):
        posts = self._db.get_posts_without_retweet_connections()
        post_ids = [p.post_osn_id for p in posts]
        are_user_ids = True
        author_type = "retweeter"
        bad_actors_collector_inseration_type = "retweeter"
        verbose_step = 15
        start = timeit.default_timer()
        post_count = len(posts)
        deleted_posts = []
        limit = 10000
        for i, post in enumerate(posts):
            try:
                retweets = self._social_network_crawler.get_retweets_by_post_id(post.post_osn_id)
                retweet_ids = []
                authors = []
                ret_posts = []
                for retweet in retweets:
                    ret_post, author = self._db._convert_tweet_to_post_and_author(retweet, self._domain)
                    ret_post.domain = 'retweet'
                    retweet_ids.append(ret_post.post_id)
                    ret_posts.append(ret_post)
                    authors.append(author)

                post_retweeter_connections = self._db.create_post_retweeter_connections(post.post_id, retweet_ids)

                self._db.add_entity_fast('posts', ret_posts)
                self._db.add_entity_fast('post_retweeter_connections', post_retweeter_connections)
                self._db.add_entity_fast('authors', authors)

                if i % verbose_step == 0:
                    end = timeit.default_timer()
                    duration = end - start
                    ramaining_time = ((float(post_count - i) / float(i + 1)) * duration) / 60.0
                    print('\r fill retweet for {}/{}, estimated time {} min'.format(str(i + 1), post_count,
                                                                                    ramaining_time),
                          end='')
            except Exception as e:
                print(e)
                deleted_posts.append(post.post_id)

    def fill_friends_connections_among_authors(self):
        self._connection_type = "friend"
        print('load posts')
        posts = self._db.get_posts()
        posts_authors = [p.author_guid for p in posts]
        authors_by_popolarity = Counter(posts_authors).most_common(len(set(posts_authors)))
        print('load authors')
        authors = self._db.get_authors_withot_connection(self._connection_type)
        authors = [author for author in authors if author.verified == '1']
        author_osn_id_author_guid_dict = {str(author.author_osn_id): author.author_guid for author in authors}
        author_osn_id_author_dict = {author.author_osn_id: author for author in authors if author.vico_dump_insertion_date is None
                                     and author.protected == 0}
        author_guid_to_osn_dict = {author.author_guid: author.author_osn_id for author in authors if author.vico_dump_insertion_date is None
                                     and author.protected == 0}
        user_ids_to_fill_followers = list(author_osn_id_author_dict.keys())

        author_osn_id_author_guid_dict = {str(author.author_osn_id): author.author_guid for id, author in
                                          author_dict.items()}
        total_authors = len(author_dict)
        total_author_connections = []
        finished_with_authors = []
        requests_count = 0
        total_authors = len(user_ids_to_fill_followers)
        start = timeit.default_timer()

        for i, (author_guid, popularity) in enumerate(authors_by_popolarity):
            if author_guid in author_dict and author_guid not in author_with_connections and popularity < 92:
                author = author_dict[author_guid]
                user_id = author.author_osn_id

                end = timeit.default_timer()
                duration = end - start
                ramaining_time = ((float(total_authors - i) / float(i + 1)) * duration) / 60.0
                print("\rSearching friends for author_guid: {0} {1}/{2}, popularity: {3}, remaining time {4} min".format(
                    user_id, i, total_authors, popularity, ramaining_time), end='')
                # author = author_osn_id_author_dict[user_id]
                insertion_date = str(get_current_time_as_string())
                author.vico_dump_insertion_date = insertion_date
                finished_with_authors.append(author)
                if requests_count < 14:
                    try:
                        author_connections = self._collect_friends_and_save_connections(user_id,
                                                                                        author_osn_id_author_guid_dict)
                        total_author_connections += author_connections
                    except Exception as e:
                        print(e)
                else:
                    requests_count = 0
                    self._db.add_author_connections_fast(total_author_connections)
                    total_author_connections = []

                if len(total_author_connections) >= 1000:
                    self._db.add_author_connections_fast(total_author_connections)
                    total_author_connections = []

                requests_count += 1
        self._db.add_author_connections_fast(total_author_connections)
        total_author_connections = []
        # self._social_network_crawler.save_author_connections()


    def fill_friends_connections_among_authors_from_csv(self):
        self._connection_type = "friend"
        # authors_by_popolarity = self._db.get_authors_by_popularity()
        print('load authors')
        # author_dict = self._db.authors_dict_by_field()

        authors = self._db.get_authors_withot_connection('friend')
        authors_df = pd.read_csv('data/output/TableToCsvExporter/authors.csv')

        # author_guid_to_osn = dict(*authors_df[['author_guid', 'author_osn_id']].itertuples(index=False))
        author_osn_to_guid = dict(authors_df[['author_osn_id', 'author_guid']].itertuples(index=False))

        author_osn_id_author_guid_dict = author_osn_to_guid

        total_authors = len(authors)
        total_author_connections = []
        finished_with_authors = []
        requests_count = 0

        for i, (author_guid, popularity) in enumerate(authors_by_popolarity):
            if author_guid in author_guid_to_osn_dict:
                user_id = author_guid_to_osn_dict[author_guid]
                print("\rSearching friends for author_guid: {0} {1}/{2}, popularity: {3}".format(user_id, i, total_authors, popularity), end='')
                author = author_osn_id_author_dict[user_id]
                insertion_date = str(get_current_time_as_string())
                author.vico_dump_insertion_date = insertion_date
                finished_with_authors.append(author)
                if requests_count < 14:
                    try:
                        author_connections = self._collect_friends_and_save_connections(user_id, author_osn_id_author_guid_dict)
                        total_author_connections += author_connections
                    except Exception as e:
                        print(e)
                else:
                    requests_count = 0
                    self._db.addPosts(total_author_connections)
                    total_author_connections = []

            if len(total_author_connections) >= 1000:
                self._db.add_author_connections_fast(total_author_connections)
                total_author_connections = []

                requests_count += 1
        self._db.add_author_connections_fast(total_author_connections)
        total_author_connections = []
        # self._social_network_crawler.save_author_connections()

    def _collect_friends_and_save_connections(self, user_id, author_osn_id_author_guid_dict):
        # user_guid = author_osn_id_author_guid_dict[user_id]
        # user_set = author_osn_id_author_guid_dict.keys()
        author_connections = self._social_network_crawler.get_friends_filtered_by_set_of_users(user_id,
                                                                                               author_osn_id_author_guid_dict)
        return author_connections

    def _collect_follower_ids_for_user_and_save_connections(self, user_id, author_osn_id_author_guid_dict):
        author_connections = []
        try:
            author_connections = self._collect_followers_and_save_connections(user_id, author_osn_id_author_guid_dict)

        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            print("Reconnecting ...")
            self._social_network_crawler = SocialNetworkCrawler(self._db)

            author_connections = self._collect_followers_and_save_connections(user_id, author_osn_id_author_guid_dict)

        except TwitterError as e:
            if e.message == "Not authorized.":
                logging.info("Not authorized for user id: {0}".format(user_id))

        return author_connections

    def _collect_followers_and_save_connections(self, user_id, author_osn_id_author_guid_dict):
        author_guid = author_osn_id_author_guid_dict[user_id]
        author_connections = []
        try:
            follower_ids = self._social_network_crawler.get_follower_ids(user_id)
            insertion_date = str(get_current_time_as_string())

            for follower_id in follower_ids:
                follower_id_str = str(follower_id)
                if follower_id_str in author_osn_id_author_guid_dict:
                    follower_author_guid = author_osn_id_author_guid_dict[follower_id_str]

                    author_connection = self._db.create_author_connection(author_guid, follower_author_guid, 0,
                                                                          self._connection_type, insertion_date)

                    author_connections.append(author_connection)
        except TwitterError as e:
            if e.message == "Not authorized.":
                logging.info("Not authorized for user id: {0}".format(user_id))

        return author_connections
