#
# Created by Aviad on 20-May-16 10:31 AM.
#


import timeit

from preprocessing_tools.abstract_controller import AbstractController
from Twitter_API.twitter_api_requester import TwitterApiRequester
from DB.schema_definition import DB, AuthorConnection, PostRetweeterConnection, Post
from commons.consts import *
from commons.commons import *
from twitter import TwitterError


class Twitter_Rest_Api(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)

        self._working_app_number = self._config_parser.eval(self.__class__.__name__, "working_app_number")

        self._maximal_get_friend_ids_requests_in_window = self._config_parser.eval(self.__class__.__name__,
                                                                                   "maximal_get_friend_ids_requests_in_window")

        self._maximal_get_follower_ids_requests_in_window = self._config_parser.eval(self.__class__.__name__,
                                                                                     "maximal_get_follower_ids_requests_in_window")

        self._maximal_get_user_requests_in_window = self._config_parser.eval(self.__class__.__name__,
                                                                             "maximal_get_user_requests_in_window")

        self._maximal_user_ids_allowed_in_single_get_user_request = self._config_parser.eval(self.__class__.__name__,
                                                                                             "maximal_user_ids_allowed_in_single_get_user_request")

        self._num_of_twitter_status_id_requests_without_checking = self._config_parser.eval(self.__class__.__name__,
                                                                                            "num_of_twitter_status_id_requests_without_checking")
        self._num_of_twitter_timeline_requests_without_checking = self._config_parser.eval(self.__class__.__name__,
                                                                                           "num_of_twitter_timeline_requests_without_checking")

        self._max_tweet_ids_allowed_in_single_get_tweets_by_tweet_ids_request = self._config_parser.eval(
            self.__class__.__name__,
            "max_tweet_ids_allowed_in_single_get_tweets_by_tweet_ids_request")

        self._max_num_of_tweet_ids_requests_without_checking = self._config_parser.eval(self.__class__.__name__,
                                                                                        "max_num_of_tweet_ids_requests_without_checking")

        self._num_of_get_friend_ids_requests = 0
        self._num_of_get_follower_ids_requests = 0
        self._num_of_get_timeline_statuses = 0
        self._num_of_twitter_status_id_requests = 0
        self._num_of_twitter_timeline_requests = 0
        self._num_of_get_tweet_ids_requests = 0
        self._total_author_connections = []

        print("Creating TwitterApiRequester")
        self._twitter_api_requester = TwitterApiRequester(self._working_app_number)

        # self._find_source_twitter_id()

        logging.info("Setup DB...")
        print("Setup DB...")
        self._db = DB()
        self._db.setUp()

    def get_timeline_by_user_id(self, user_id):
        try:
            if self._num_of_get_timeline_statuses > self._num_of_twitter_timeline_requests_without_checking:
                seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_timeline()
                if seconds_to_wait != 0:
                    self.count_down_time(seconds_to_wait)
                    self._num_of_get_timeline_statuses = 0
                timeline = self._twitter_api_requester.get_timeline_by_user_id(user_id)
                self._num_of_get_timeline_statuses += 1
                print("Number of get timeline requests is: " + str(self._num_of_get_timeline_statuses))

            return timeline

        except TwitterError as e:
            logging.info(e.message)
            if e.message == "Not authorized.":
                logging.info("Not authorized for user id: " + str(user_id))
                return None
            sec = self._twitter_api_requester.get_sleep_time_for_timeline()
            logging.info("Seconds to wait from catched crush is: " + str(sec))
            count_down_time(sec)
            self._num_of_get_timeline_statuses = 0
            timeline = self._twitter_api_requester.get_timeline_by_user_id(user_id)
            return timeline

    def handle_get_follower_ids_request(self, source_id):
        print("--- handle_get_follower_ids_request ---")
        logging.info("--- handle_get_follower_ids_request ---")
        follower_ids = self._twitter_api_requester.get_follower_ids_by_user_id(source_id)
        follower_connection_type = str(Author_Connection_Type.FOLLOWER)
        temp_author_connections = self._db.create_temp_author_connections(source_id, follower_ids,
                                                                          follower_connection_type)
        self._total_author_connections = self._total_author_connections + temp_author_connections
        return follower_ids

    def handle_get_user_ids_request(self, source_id, author_type):
        print("--- handle_get_user_ids_request ---")
        if author_type == Author_Connection_Type.FOLLOWER:
            user_ids = self._twitter_api_requester.get_follower_ids_by_user_id(source_id)
        elif author_type == Author_Connection_Type.FRIEND:
            user_ids = self._twitter_api_requester.get_friend_ids_by_user_id(source_id)

        author_connections = self.create_author_connections(source_id, user_ids, author_type)
        self._total_author_connections = self._total_author_connections + author_connections
        return user_ids

    def handle_get_friend_ids_request(self, source_id):
        friend_ids = self._twitter_api_requester.get_friend_ids_by_user_id(source_id)
        friend_connection_type = str(Author_Connection_Type.FRIEND)
        author_connections = self.create_author_connections(source_id, friend_ids, friend_connection_type)
        self._total_author_connections = self._total_author_connections + author_connections
        return friend_ids

    def get_friends_filtered_by_set_of_users(self, source_id, author_osn_id_author_guid_dict):
        source_guid = author_osn_id_author_guid_dict[source_id]
        user_set = list(author_osn_id_author_guid_dict.keys())
        if user_set:
            friend_ids = self._twitter_api_requester.get_friend_ids_by_user_id(source_id)
            friend_ids = list(set(map(str, friend_ids)) & set(map(str,user_set)))
            friend_guids = list(map(author_osn_id_author_guid_dict.get, list(map(int, friend_ids))))
            friend_connection_type = str(Author_Connection_Type.FRIEND)
            author_connections = self.create_author_connections(source_guid, friend_guids, friend_connection_type)
            return author_connections
        else:
            return []

    def crawl_users_by_author_ids(self, author_ids, connection_type, author_type, are_user_ids, insertion_type):
        self._total_author_connections = []

        total_user_ids = self.crawl_users(author_ids, connection_type)

        self._db.save_author_connections(self._total_author_connections)

        total_user_ids_to_crawl = self.remove_already_crawled_authors(total_user_ids)

        users = self.handle_get_users_request(total_user_ids_to_crawl, are_user_ids, author_type, insertion_type)
        self.convert_twitter_users_to_authors_and_save(users, author_type, insertion_type)

    def crawl_users(self, author_ids, author_type):
        print("--- crawl_users ---")
        total_user_ids = []
        for author_id in author_ids:
            print("--- crawl_user_ids for author id : " + str(author_id))

            get_sleep_function_name = "get_sleep_time_for_get_" + author_type + "_ids_request"
            seconds_to_wait = getattr(self._twitter_api_requester, get_sleep_function_name)()
            if seconds_to_wait != 0:
                self.save_connections_and_wait(seconds_to_wait)
                init_num_of_get_user_ids_requests_func_name = "init_num_of_get_" + author_type + "_ids_requests"
                getattr(self._twitter_api_requester, init_num_of_get_user_ids_requests_func_name)()

            get_user_ids_by_given_user_id_function_name = "get_" + author_type + "_ids_by_user_id"
            user_ids = getattr(self._twitter_api_requester, get_user_ids_by_given_user_id_function_name)(author_id)

            temp_author_connections = self._db.create_temp_author_connections(author_id, user_ids, author_type,
                                                                              self._window_start)
            self._total_author_connections = self._total_author_connections + temp_author_connections

            total_user_ids = total_user_ids + user_ids
            #total_user_ids = list(set(total_user_ids))

        return total_user_ids

    def check_already_crawled_author_guids(self, author_guids):
        print("--- check_already_crawled_author_ids ----")
        author_ids_to_crawl = []
        for author_guid in author_guids:
            authors_connections = self._db.get_author_connections_by_author_guid(author_guid)
            num_of_authors_connections = len(authors_connections)
            if num_of_authors_connections == 0:
                author_ids_to_crawl.append(author_guid)

        print("Number of authors ids to crawl is: " + str(len(author_ids_to_crawl)))
        logging.info("Number of authors ids to crawl is: " + str(len(author_ids_to_crawl)))
        print(author_ids_to_crawl)
        logging.info(author_ids_to_crawl)
        return author_ids_to_crawl

    def check_already_crawled_post_id(self, post_id):
        post_retweeter_connections = self._db.get_post_retweeter_connections_by_post_id(post_id)
        num_of_post_retweeter_connections = len(post_retweeter_connections)
        if num_of_post_retweeter_connections == 0:
            return False
        return True

    def crawl_retweeters_by_post_id(self, post_ids, are_user_ids, author_type, bad_actors_collector_inseration_type):

        self._total_author_connections = []
        total_retweeter_ids = []
        start = timeit.default_timer()
        post_count = len(post_ids)
        deleted_posts = []
        verbose_step = 15
        for i, post_id in enumerate(post_ids):
            try:
                retweeter_ids = self._twitter_api_requester.get_retweeter_ids_by_status_id(post_id)
                total_retweeter_ids = list(set(total_retweeter_ids + retweeter_ids))

                post_retweeter_connections = self._db.create_post_retweeter_connections(post_id, retweeter_ids)
                self._total_author_connections = self._total_author_connections + post_retweeter_connections

                if i % verbose_step == 0:
                    end = timeit.default_timer()
                    duration = end - start
                    ramaining_time = (((post_count - i) / verbose_step) * duration) / 60
                    print('\r fill retweet for {}/{}, estimated time {} min'.format(str(i+1), post_count, ramaining_time), end='')
            except TwitterError as e:
                print(e)
                deleted_posts.append(post_id)
        print()
        print('posts deleted')
        self._db.save_author_connections(self._total_author_connections)
        self._total_author_connections = []

        users = self.handle_get_users_request(total_retweeter_ids, are_user_ids, author_type,
                                              bad_actors_collector_inseration_type)
        self.convert_twitter_users_to_authors_and_save(users, author_type, bad_actors_collector_inseration_type)

    def get_retweets_by_post_id(self, post_id):
        try:
            retweets = self._twitter_api_requester.get_retweets_by_status_id(post_id)
            return retweets
        except TwitterError as e:
                print(e)
                error_messages = e.message
                error_message_dict = error_messages[0]
                error_code = error_message_dict['code']
                if error_code == 34:  # 'Sorry, that page does not exist.'
                    return []
                if error_code == 88:  # Rate limit exceeded
                    seconds_to_wait_object = self._twitter_api_requester.get_sleep_time_for_get_retweets_request()
                    if seconds_to_wait_object > 0:
                        count_down_time(seconds_to_wait_object)
                    return self._twitter_api_requester.get_retweets_by_status_id(post_id)
                if error_code == 200:  # forbidden
                    return []

    def create_author_connections(self, source_author_id, destination_author_ids, author_connection_type):
        print("---create_author_connections---")
        logging.info("---create_author_connections---")
        author_connections = []
        for destination_author_id in destination_author_ids:
            author_connection = self._db.create_author_connection(source_author_id, destination_author_id, 0.0,
                                                                  author_connection_type, None)
            author_connections.append(author_connection)

        return author_connections

    def create_author_connection(self, source_author_id, destination_author_id, connection_type):
        # print("---create_author_connection---")
        author_connection = AuthorConnection()
        # print("Author connection: source -> " + str(source_author_id) + ", dest -> " + str(
        #     destination_author_id) + ", connection type = " + connection_type)
        author_connection.source_author_guid = str(source_author_id)
        author_connection.destination_author_guid = str(destination_author_id)
        author_connection.connection_type = str(connection_type)
        author_connection.insertion_date = self._window_start

        return author_connection

    def count_down_time(self, seconds_to_wait):
        if seconds_to_wait is not 0:
            print("Seconds to wait is lower than 300: " + str(seconds_to_wait))
            logging.info("Seconds to wait is lower than 300: " + str(seconds_to_wait))
            seconds_to_wait += 100
            print("Seconds to wait were increased to: " + str(seconds_to_wait))
            logging.info("Seconds to wait were increased to: " + str(seconds_to_wait))
        elif seconds_to_wait is not 0 and seconds_to_wait < 400:
            print("Seconds to wait is lower than 400: " + str(seconds_to_wait))
            logging.info("Seconds to wait is lower than 400: " + str(seconds_to_wait))
            seconds_to_wait += 90
            print("Seconds to wait were increased to: " + str(seconds_to_wait))
            logging.info("Seconds to wait were increased to: " + str(seconds_to_wait))
        for i in range(seconds_to_wait, 0, -1):
            time.sleep(1)
            msg = "\r Count down: [{}]".format(i)
            print(msg, end="")
            # sys.stdout.write(str(i)+' ')
            # sys.stdout.flush()

    def convert_twitter_users_to_authors_and_save(self, total_twitter_users, author_type, inseration_type):
        authors = self.convert_twitter_users_to_authors(total_twitter_users, author_type, inseration_type)
        print("Total converted Twitter users into authors is: " + str(len(authors)))
        self.save_authors(authors)
        self._db.save_author_connections(self._total_author_connections)
        self._db.session.commit()
        self._total_author_connections = []

        return authors

    def convert_twitter_users_to_authors(self, total_twitter_users, author_type, inseration_type):
        print("---Converting Twitter users to authors---")
        convert_twitter_users_to_authors_start_time = time.time()
        authors = self._db.convert_twitter_users_to_authors(total_twitter_users, self._domain, author_type,
                                                            inseration_type)
        convert_twitter_users_to_authors_end_time = time.time()
        convert_twitter_users_to_authors_time = convert_twitter_users_to_authors_end_time - convert_twitter_users_to_authors_start_time
        print("Convert Twitter users to authors took in seconds: " + str(convert_twitter_users_to_authors_time))

        return authors

    def save_authors(self, authors):
        print("---Saving authors in DB---")
        print("Number of authors to save is: " + str(len(authors)))
        save_authors_start_time = time.time()
        self._db.insert_or_update(authors)
        # self._db.add_authors_fast(authors)
        save_authors_end_time = time.time()
        save_authors_time = save_authors_end_time - save_authors_start_time
        print("Saving authors in DB took in seconds: " + str(save_authors_time))

    def save_author_connections(self):
        print("---Saving author connections in DB---")
        save_author_connections_start_time = time.time()
        self._db.add_author_connections(self._total_author_connections)
        save_author_connections_end_time = time.time()
        save_author_connections_time = save_author_connections_end_time - save_author_connections_start_time
        print("Saving author connections in DB took in seconds: " + str(save_author_connections_time))
        self._total_author_connections = []

    def handle_get_users_request(self, ids, are_user_ids, author_type, insertion_type):
        total_users = []
        users = []
        ids_in_chunks = split_into_equal_chunks(ids,
                                                self._maximal_user_ids_allowed_in_single_get_user_request)
        total_chunks = list(ids_in_chunks)
        ids_in_chunks = split_into_equal_chunks(ids,
                                                self._maximal_user_ids_allowed_in_single_get_user_request)
        print("Total authors ids in chunk from twitter API: " + str(len(total_chunks)))
        i = 0
        for ids_in_chunk in ids_in_chunks:
            i += 1
            print("Chunk of authors ids: " + str(i) + "/" + str(len(total_chunks)))

            try:
                users = self.send_get_users_request_and_add_users(ids_in_chunk, are_user_ids,
                                                                  users)
                total_users = total_users + users

            except Exception as e:

                print(e)
                error_messages = e.message
                error_message_dict = error_messages[0]
                error_code = error_message_dict['code']
                if error_code == 88:  # Rate limit exceeded
                    self.convert_twitter_users_to_authors_and_save(total_users, author_type, insertion_type)
                    total_users = []

                    seconds_to_wait_object = self._twitter_api_requester.get_sleep_time_for_get_users_request()
                    if seconds_to_wait_object > 0:
                        count_down_time(seconds_to_wait_object)
                    #epoch_timestamp = seconds_to_wait_object.reset
                    #current_timestamp = time.time()
                    #seconds_to_wait = int(epoch_timestamp - current_timestamp + 5)
                    #count_down_time(seconds_to_wait)

                    users = self.send_get_users_request_and_add_users(ids_in_chunk, are_user_ids,
                                                                      users)
                    total_users = total_users + users

        print("--- Finishing handle_get_users_request --- ")
        logging.info("--- Finishing handle_get_users_request --- ")
        # self.save_authors_and_connections(users, author_type, insertion_type)
        return total_users

    def save_authors_and_connections_and_wait(self, total_twitter_users, author_type, inseration_type):
        self.save_authors_and_connections(total_twitter_users, author_type, inseration_type)

        seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_get_users_request()
        self.count_down_time(seconds_to_wait)

    def save_authors_and_connections(self, total_twitter_users, author_type, inseration_type):
        authors = self.convert_twitter_users_to_authors_and_save(total_twitter_users, author_type, inseration_type)
        return authors

    def send_get_users_request_and_add_users(self, ids_in_chunk, are_user_ids, total_twitter_users):
        twitter_users = self.send_get_users_request(ids_in_chunk, are_user_ids)
        return twitter_users

    def save_connections_and_wait(self, seconds_to_wait):
        self.save_author_connections()
        self.count_down_time(seconds_to_wait)

    def send_get_users_request(self, ids_in_chunk, are_user_ids):
        if are_user_ids is True:
            twitter_users = self._twitter_api_requester.get_users_by_ids(ids_in_chunk)
        else:
            twitter_users = self._twitter_api_requester.get_users_by_screen_names(ids_in_chunk)

        return twitter_users

    def handle_retweeters_request(self, retweeter_ids, author_type, bad_actors_collector_inseration_type):
        total_retweeters = []
        retweeter_ids_in_chunks = split_into_equal_chunks(retweeter_ids,
                                                          self._maximal_user_ids_allowed_in_single_get_user_request)
        for retweeter_ids_in_chunk in retweeter_ids_in_chunks:
            retweeters = self._twitter_api_requester.get_users_by_ids(retweeter_ids_in_chunk)
            total_retweeters = total_retweeters + retweeters

        self.convert_twitter_users_to_authors_and_save(total_retweeters, author_type,
                                                       bad_actors_collector_inseration_type)

    def remove_already_crawled_authors(self, total_user_ids):
        print("remove_already_crawled_authors")
        number_of_extracted_users = len(total_user_ids)
        print("Total number of extracted users is: " + str(number_of_extracted_users))
        total_follower_ids_set = set(total_user_ids)

        already_crawled_author_ids = self._db.get_already_crawled_author_ids()
        number_of_already_crawled_authors = len(already_crawled_author_ids)
        print("Total number of already crawled users is: " + str(number_of_already_crawled_authors))
        already_crawled_author_ids_set = set(already_crawled_author_ids)

        authors_ids_to_crawl_set = total_follower_ids_set - already_crawled_author_ids_set
        number_of_remaining_authors_ids_to_crawl = len(authors_ids_to_crawl_set)
        print("Total number of remaining users to crawl is: " + str(number_of_remaining_authors_ids_to_crawl))

        authors_ids_to_crawl = list(authors_ids_to_crawl_set)

        return authors_ids_to_crawl

    def get_timline_by_author_id(self, author_id):
        author_timeline = self._twitter_api_requester.get_timeline_by_user_id(author_id)
        return author_timeline

    def get_status_by_twitter_status_id(self, id):
        # try:
        if self._num_of_twitter_status_id_requests >= self._num_of_twitter_status_id_requests_without_checking:
            seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_twitter_status_id()
            if seconds_to_wait > 0:
                self.count_down_time(seconds_to_wait)
                self._num_of_twitter_status_id_requests = 0
        self._num_of_twitter_status_id_requests = self._num_of_twitter_status_id_requests + 1
        return self._twitter_api_requester.get_status(id)
        # except TwitterError as e:
        #     exception_response = e[0][0]
        #     logging.info("e.massage =" + exception_response["message"])
        #     code = exception_response["code"]
        #     logging.info("e.code =" + str(exception_response["code"]))
        #
        #     if code == 88:
        #         sec = self._twitter_api_requester.get_sleep_time_for_twitter_status_id()
        #         logging.info("Seconds to wait from catched crush is: " + str(sec))
        #         if sec != 0:
        #             count_down_time(sec)
        #             self._num_of_twitter_status_id_requests = 0
        #         return self._twitter_api_requester.get_status(id)

    def get_timeline_by_author_name(self, author_name, maximal_tweets_count_in_timeline):
        try:
            print("Number of timeline requests is: " + str(self._num_of_twitter_timeline_requests))
            if self._num_of_twitter_timeline_requests >= self._num_of_twitter_timeline_requests_without_checking:
                seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_twitter_timeline_request()
                if seconds_to_wait > 0:
                    self.count_down_time(seconds_to_wait)
                    self._num_of_twitter_timeline_requests = 0
            self._num_of_twitter_timeline_requests = self._num_of_twitter_timeline_requests + 1
            return self._twitter_api_requester.get_timeline(author_name, maximal_tweets_count_in_timeline)

        except TwitterError as e:
            if e.message == "Not authorized.":
                logging.info("Not authorized for user id: " + str(author_name))
                return None

            exception_response = e[0][0]
            logging.info("e.massage =" + exception_response["message"])
            code = exception_response["code"]
            logging.info("e.code =" + str(exception_response["code"]))

            if code == 34:
                return None

            sec = self._twitter_api_requester.get_sleep_time_for_twitter_timeline_request()
            logging.info("Seconds to wait from catched crush is: " + str(sec))
            count_down_time(sec)
            if sec != 0:
                self._num_of_twitter_timeline_requests = 0
            timeline = self._twitter_api_requester.get_timeline(author_name, maximal_tweets_count_in_timeline)
            return timeline

    def get_timeline(self, author_name, maximal_tweets_count_in_timeline):
        timeline = self._twitter_api_requester.get_timeline(author_name, maximal_tweets_count_in_timeline)
        return timeline

    def get_active_users_names_by_screen_names(self, chunk_of_names):
        try:
            users = self._twitter_api_requester.get_users_by_screen_names(chunk_of_names)
        except TwitterError as e:
            logging.info(e.message)
            sec = self._twitter_api_requester.get_sleep_time_for_get_users_request()
            logging.info("Seconds to wait from catched crush is: " + str(sec))
            count_down_time(sec)
            users = self._twitter_api_requester.get_users_by_screen_names(chunk_of_names)
        return [user.screen_name for user in users]

    def get_sleep_time_for_twitter_status_id(self):
        return self._twitter_api_requester.get_sleep_time_for_twitter_status_id()

    def get_status(self, id):

        return self._twitter_api_requester.get_status(id)

    def get_posts_by_terms(self, terms):
        posts = {term: self._twitter_api_requester.get_tweets_by_term(term, 'recent') for term in terms}
        return posts

    def get_post_by_post_id(self, post_id):
        return self._twitter_api_requester.get_tweet_by_post_id(post_id)

    def get_tweets_by_tweet_ids_and_add_to_db(self, tweet_ids):
        total_tweets = self.get_tweets_by_ids(tweet_ids)

        posts, authors = self._db.convert_tweets_to_posts_and_authors(total_tweets, self._domain)
        self._db.addPosts(posts)
        self._db.add_authors(authors)

        return total_tweets

        # move to schema definition

    def get_tweets_by_ids(self, tweet_ids, author_type=""):
        total_tweets = []
        ids_in_chunks = split_into_equal_chunks(tweet_ids,
                                                self._max_tweet_ids_allowed_in_single_get_tweets_by_tweet_ids_request)
        # seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_get_tweets_by_tweet_ids_request()
        total_chunks = list(ids_in_chunks)
        ids_in_chunks = split_into_equal_chunks(tweet_ids,
                                                self._max_tweet_ids_allowed_in_single_get_tweets_by_tweet_ids_request)
        i = 0
        for ids_in_chunk in ids_in_chunks:
            i += 1
            print("Chunk of tweet ids: " + str(i) + "/" + str(len(total_chunks)))
            try:
                tweets = self._twitter_api_requester.get_tweets_by_post_ids(ids_in_chunk)
                total_tweets = list(set(total_tweets + tweets))
                num_of_tweets = len(total_tweets)
                if num_of_tweets > 10000:
                    self._save_posts_and_authors(total_tweets, author_type)
                    total_tweets = []

            except TwitterError as e:
                print(e)
                error_messages = e.message
                error_message_dict = error_messages[0]
                error_code = error_message_dict['code']
                if error_code == 88:  # Rate limit exceeded
                    self._save_posts_and_authors(total_tweets, author_type)
                    total_tweets = []

                    seconds_to_wait_object = self._twitter_api_requester.get_sleep_time_for_get_tweets_by_tweet_ids_request()
                    epoch_timestamp = seconds_to_wait_object.reset
                    current_timestamp = time.time()
                    seconds_to_wait = int(epoch_timestamp - current_timestamp + 5)
                    count_down_time(seconds_to_wait)
                    tweets = self._twitter_api_requester.get_tweets_by_post_ids(ids_in_chunk)
                    total_tweets = list(set(total_tweets + tweets))
        return total_tweets

    # def create_post_from_tweet_data(self, tweet_data):
    #     author_name = tweet_data.user.screen_name
    #     tweet_author_guid = compute_author_guid_by_author_name(author_name)
    #     tweet_author_guid = cleanForAuthor(tweet_author_guid)
    #     tweet_post_twitter_id = str(tweet_data.id)
    #     tweet_url = generate_tweet_url(tweet_post_twitter_id, author_name)
    #     tweet_creation_time = tweet_data.created_at
    #     tweet_str_publication_date = extract_tweet_publiction_date(tweet_creation_time)
    #     tweet_guid = compute_post_guid(post_url=tweet_url, author_name=author_name,
    #                                    str_publication_date=tweet_str_publication_date)
    #
    #     post = Post(guid=tweet_guid, post_id=tweet_guid, url=unicode(tweet_url),
    #                       date=str_to_date(tweet_str_publication_date),
    #                       title=unicode(tweet_data.text), content=unicode(tweet_data.text),
    #                       post_osn_id=tweet_post_twitter_id,
    #                       author=unicode(author_name), author_guid=unicode(tweet_author_guid),
    #                       domain=unicode(self._domain),
    #                       retweet_count=unicode(tweet_data.retweet_count),
    #                       favorite_count=unicode(tweet_data.favorite_count),
    #                       timeline_importer_insertion_date=unicode(get_current_time_as_string()))
    #     return post

    def _save_posts_and_authors(self, total_tweets, author_type=None):
        posts, authors = self._db.convert_tweets_to_posts_and_authors(total_tweets, self._domain)
        for author in authors:
            author.author_type = author_type
        self._db.addPosts(posts)
        self._db.addPosts(authors)
