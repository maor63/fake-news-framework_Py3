#
# Created by Aviad on 20-May-16 10:31 AM.
#

from preprocessing_tools.abstract_controller import AbstractController
from Twitter_API.twitter_api_requester import TwitterApiRequester
from DB.schema_definition import DB, AuthorConnection, PostRetweeterConnection
from commons.consts import *
from commons.commons import *
from twitter import TwitterError


class SocialNetworkCrawler(AbstractController):
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

        self._num_of_get_friend_ids_requests = 0
        self._num_of_get_follower_ids_requests = 0
        self._num_of_get_timeline_statuses = 0
        self._num_of_twitter_status_id_requests = 0
        self._num_of_twitter_timeline_requests = 0
        self._num_of_get_twitter_users_requests = 0
        self._total_author_connections = []
        self._total_follower_ids = []

        print("Creating TwitterApiRequester")
        self._twitter_api_requester = TwitterApiRequester(self._working_app_number)

        # self._find_source_twitter_id()

        logging.info("Setup DB...")
        print("Setup DB...")
        self._db = DB()
        self._db.setUp()

    def fill_followers_ids_only(self, author_ids):
        for i, author_id in enumerate(author_ids):
            print("author_id: {0} {1}/{2}".format(author_id, i, len(author_ids)))
            follower_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)
            temp_author_connections = self._db.create_temp_author_connections(author_id, follower_ids, "follower",
                                                                              self._window_start)
            self._total_author_connections = self._total_author_connections + temp_author_connections

            if len(self._total_author_connections) > 1000000:
                self._db.addPosts(self._total_author_connections)
                self._total_author_connections = []

        self._db.addPosts(self._total_author_connections)

    def fill_followers_and_their_data_simultaneously(self, author_ids):

        for i, author_id in enumerate(author_ids):
            if self._num_of_get_follower_ids_requests < self._maximal_get_follower_ids_requests_in_window:

                self._send_get_follower_ids_for_author_id(author_id, i, author_ids)

            else:
                author_type = None
                are_user_ids = True
                insertion_type = DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR
                users = self.handle_get_users_request(self._total_follower_ids, are_user_ids, author_type,
                                                      insertion_type)
                self.convert_twitter_users_to_authors_and_save(users, "follower", insertion_type)
                #self._db.addPosts(self._total_author_connections)
                #self._db.save_author_connections(self._total_author_connections)

                self._num_of_get_twitter_users_requests = 0
                self._total_follower_ids = []
                #self._total_author_connections = []

                T = time.time()
                diff = (T - self._last_follower_request_time) / 60
                # window time to wait
                if diff < 15:
                    count_down_time(diff * 60)
                self._send_get_follower_ids_for_author_id(author_id, i, author_ids)


    def _send_get_follower_ids_for_author_id(self, author_id, i, author_ids):
        follower_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)

        self._last_follower_request_time = time.time()

        self._num_of_get_follower_ids_requests += 1

        print("Bring followers {0}:{1}/{2}".format(author_id, i, len(author_ids)))

        self._total_follower_ids = self._total_follower_ids + follower_ids

        temp_author_connections = self._db.create_temp_author_connections(author_id, follower_ids, "follower",
                                                                          self._window_start)
        self._total_author_connections = self._total_author_connections + temp_author_connections



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

    def crawl_users_by_author_ids(self, author_ids, connection_type, author_type, are_user_ids, insertion_type):
        self._total_author_connections = []

        total_follower_ids, already_checked_author_ids = self.get_followers_until_exception(author_ids, connection_type)

        self._db.save_author_connections(self._total_author_connections)

        total_user_ids_to_crawl = self.remove_already_crawled_authors(total_follower_ids)

        users = self.handle_get_users_request(total_user_ids_to_crawl, are_user_ids, author_type, insertion_type)
        self.convert_twitter_users_to_authors_and_save(users, author_type, insertion_type)

        return total_follower_ids, already_checked_author_ids


    def get_follower_ids(self, author_id):
        user_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)
        return user_ids

    def get_sleep_time_for_follower_ids(self):
        user_ids = self._twitter_api_requester.get_sleep_time_for_follower_ids()
        return user_ids

    # def crawl_followers_by_twitter_author_ids(self, author_ids, author_type, are_user_ids, inseration_type):
    #     print("--- crawl_followers_by_twitter_author_ids ---")
    #
    #     #authors_ids_to_crawl = self.check_already_crawled_author_ids(author_ids)
    #     total_follower_ids = self.crawl_followers_ids(author_ids)
    #
    #     self.save_author_connections()
    #
    #     total_follower_ids_to_crawl = self.remove_already_crawled_authors(total_follower_ids)
    #
    #     self.handle_get_users_request(total_follower_ids_to_crawl, are_user_ids, author_type, inseration_type)
    #     #self.convert_twitter_users_to_authors_and_save(followers, author_type, inseration_type)

    # def crawl_friends_by_twitter_author_ids(self, author_ids, author_type, are_user_ids, inseration_type):
    #     # authors_ids_to_crawl = self.check_already_crawled_author_ids(author_ids)
    #     total_friends_ids = self.crawl_friends_ids(author_ids)
    #
    #     self.save_author_connections()
    #
    #     total_friends_ids_to_crawl = self.remove_already_crawled_authors(total_friends_ids)
    #
    #     friends = self.handle_get_users_request(total_friends_ids_to_crawl, are_user_ids, author_type, inseration_type)
    #     self.convert_twitter_users_to_authors_and_save(friends, author_type, inseration_type)

    # def crawl_retweeters_by_twitter_post_ids(self, post_ids, author_type, inseration_type):
    #     #authors_ids_to_crawl = self.check_already_crawled_author_ids(post_ids)
    #     total_follower_ids = self.crawl_retweeters_ids(post_ids)
    #
    #     self.save_author_connections()
    #     are_user_ids = True
    #     followers = self.handle_get_users_request(total_follower_ids, are_user_ids, author_type, inseration_type)
    #     self.convert_twitter_users_to_authors_and_save(followers, author_type, inseration_type)

    # def crawl_retweeters_ids(self, posts_ids):
    #     total_retweeter_ids = []
    #     for posts_id in posts_ids:
    #         seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_get_retweeter_ids_request()
    #         if seconds_to_wait == 0:
    #             retweeter_ids = self.handle_get_retweeter_ids_request(posts_id)
    #             total_retweeter_ids = list(set(total_retweeter_ids + retweeter_ids))
    #         else:
    #             self.save_connections_and_wait(seconds_to_wait)
    #             self._twitter_api_requester.init_num_of_get_follower_ids_requests()
    #             retweeter_ids = self.handle_get_retweeter_ids_request(posts_id)
    #             total_retweeter_ids = list(set(total_retweeter_ids + retweeter_ids))
    #     return total_retweeter_ids

    # def crawl_users(self, author_ids, author_type):
    #     print("--- crawl_users ---")
    #     total_user_ids = []
    #     for author_id in author_ids:
    #         print("--- crawl_user_ids for author id : " + str(author_id))
    #
    #         get_sleep_function_name = "get_sleep_time_for_get_" + author_type + "_ids_request"
    #         seconds_to_wait = getattr(self._twitter_api_requester, get_sleep_function_name)()
    #         if seconds_to_wait != 0:
    #             self.save_connections_and_wait(seconds_to_wait)
    #             init_num_of_get_user_ids_requests_func_name = "init_num_of_get_" + author_type + "_ids_requests"
    #             getattr(self._twitter_api_requester, init_num_of_get_user_ids_requests_func_name)()
    #
    #         get_user_ids_by_given_user_id_function_name = "get_" + author_type + "_ids_by_user_id"
    #         user_ids = getattr(self._twitter_api_requester, get_user_ids_by_given_user_id_function_name)(author_id)
    #
    #         temp_author_connections = self._db.create_temp_author_connections(author_id, user_ids, author_type,
    #                                                                           self._window_start)
    #         self._total_author_connections = self._total_author_connections + temp_author_connections
    #
    #         total_user_ids = list(set(total_user_ids + user_ids))
    #
    #     return total_user_ids

    def crawl_users(self, author_ids, author_type):
        total_user_ids = []
        for i, author_id in enumerate(author_ids):
            msg = "\r Bring followers for authors: {0}/{1}".format(i, len(author_ids))
            print(msg, end="")

            try:
                user_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)

                temp_author_connections = self._db.create_temp_author_connections(author_id, user_ids, author_type,
                                                                                  self._window_start)
                self._total_author_connections = self._total_author_connections + temp_author_connections

                total_user_ids = list(set(total_user_ids + user_ids))

            except TwitterError as e:
                exception_response = e[0][0]
                logging.info("e.massage =" + exception_response["message"])
                code = exception_response["code"]
                logging.info("e.code =" + str(exception_response["code"]))

                if code == 88:
                    sec = self._twitter_api_requester.get_sleep_time_for_get_follower_ids_request()
                    sec = sec + 100
                    logging.info("Seconds to wait from catched crush is: " + str(sec))
                    if sec != 0:
                        count_down_time(sec)

                    user_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)

                    temp_author_connections = self._db.create_temp_author_connections(author_id, user_ids, author_type,
                                                                                      self._window_start)
                    self._total_author_connections = self._total_author_connections + temp_author_connections

                    total_user_ids = list(set(total_user_ids + user_ids))

        return total_user_ids



    def get_followers_until_exception(self, author_ids, author_type):
        total_follower_ids = []
        already_checked_author_ids = []
        for i, author_id in enumerate(author_ids):
            msg = "\r Bring followers for authors: {0}/{1}".format(i, len(author_ids))
            print(msg, end="")

            try:
                user_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)
                already_checked_author_ids.append(author_id)

                if len(user_ids) > 0:
                    temp_author_connections = self._db.create_temp_author_connections(author_id, user_ids, author_type,
                                                                                      self._window_start)
                    self._total_author_connections = self._total_author_connections + temp_author_connections

                    total_follower_ids = list(set(total_follower_ids + user_ids))
            except TwitterError as e:
                if e.message == "Not authorized.":
                    logging.info("Not authorized for user id: {0}".format(author_id))
                    return total_follower_ids, already_checked_author_ids

                exception_response = e[0][0]
                logging.info("e.massage =" + exception_response["message"])
                code = exception_response["code"]
                logging.info("e.code =" + str(exception_response["code"]))

                if code == 34:
                    return total_follower_ids, already_checked_author_ids

                if code == 88:
                    sec = self._twitter_api_requester.get_sleep_time_for_get_follower_ids_request()
                    sec = sec + 10
                    # logging.info("Seconds to wait from catched crush is: " + str(sec))
                    # if sec != 0:
                    print("Number of seconds to wait: {0}".format(sec))
                    count_down_time(sec)

                    try:
                        user_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)
                        already_checked_author_ids.append(author_id)

                        temp_author_connections = self._db.create_temp_author_connections(author_id, user_ids,
                                                                                          author_type,
                                                                                          self._window_start)
                        self._total_author_connections = self._total_author_connections + temp_author_connections

                        total_follower_ids = list(set(total_follower_ids + user_ids))

                    except TwitterError as e:
                        if e.message == "Not authorized.":
                            logging.info("Not authorized for user id: {0}".format(author_id))
                            return total_follower_ids, already_checked_author_ids

            # except TwitterError as e:
            #     exception_response = e[0][0]
            #     logging.info("e.massage =" + exception_response["message"])
            #     code = exception_response["code"]
            #     logging.info("e.code =" + str(exception_response["code"]))
            #
            #     if code == 88 and len(already_checked_author_ids) != 0:
            #         return total_follower_ids, already_checked_author_ids
            #     elif code == 88 and len(already_checked_author_ids) == 0:
            #         sec = self._twitter_api_requester.get_sleep_time_for_get_follower_ids_request()
            #         sec = sec + 3100
            #         # logging.info("Seconds to wait from catched crush is: " + str(sec))
            #         # if sec != 0:
            #         print("Number of seconds to wait: {0}".format(sec))
            #         count_down_time(sec)
            #         try:
            #             user_ids = self._twitter_api_requester.get_follower_ids_by_user_id(author_id)
            #             already_checked_author_ids.append(author_id)
            #
            #             temp_author_connections = self._db.create_temp_author_connections(author_id, user_ids, author_type,
            #                                                                               self._window_start)
            #             self._total_author_connections = self._total_author_connections + temp_author_connections
            #
            #             total_follower_ids = list(set(total_follower_ids + user_ids))
            #             return total_follower_ids, already_checked_author_ids
            #         except TwitterError as e:
            #             if e.message == "Not authorized.":
            #                 logging.info("Not authorized for user id: {0}".format(author_id))
            #                 return total_follower_ids, already_checked_author_ids

        return total_follower_ids, already_checked_author_ids




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
        for post_id in post_ids:
            retweeter_ids = self._twitter_api_requester.get_retweeter_ids_by_status_id(post_id)
            total_retweeter_ids = list(set(total_retweeter_ids + retweeter_ids))

            post_retweeter_connections = self._db.create_post_retweeter_connections(post_id, retweeter_ids)
            self._total_author_connections = self._total_author_connections + post_retweeter_connections

        self._db.save_author_connections(self._total_author_connections)
        self._total_author_connections = []

        users = self.handle_get_users_request(total_retweeter_ids, are_user_ids, author_type,
                                              bad_actors_collector_inseration_type)
        self.convert_twitter_users_to_authors_and_save(users, author_type, bad_actors_collector_inseration_type)

    def get_retweets_by_post_id(self, post_id):
        retweets = self._twitter_api_requester.get_retweets_by_status_id(post_id)
        print(retweets)

    # def create_author_connections(self, source_author_id, destination_author_ids, author_connection_type):
    #     print("---create_author_connections---")
    #     logging.info("---create_author_connections---")
    #     author_connections = []
    #     for destination_author_id in destination_author_ids:
    #         author_connection = self.create_author_connection(source_author_id, destination_author_id, author_connection_type)
    #         author_connections.append(author_connection)
    #
    #     return author_connections

    # def create_author_connection(self, source_author_id, destination_author_id, connection_type):
    #     print("---create_author_connection---")
    #     author_connection = AuthorConnection()
    #     print("Author connection: source -> " + str(source_author_id) + ", dest -> " + str(destination_author_id) + ", connection type = " + connection_type)
    #     author_connection.source_author_osn_id = source_author_id
    #     author_connection.destination_author_osn_id = destination_author_id
    #     author_connection.connection_type = unicode(connection_type)
    #     author_connection.insertion_date = self._window_start
    #
    #     return author_connection

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
        self._total_author_connections = []

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
        self._db.add_authors(authors)
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
        #seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_get_users_request()
        total_chunks = list(ids_in_chunks)
        ids_in_chunks = split_into_equal_chunks(ids,
                                                self._maximal_user_ids_allowed_in_single_get_user_request)
        print("Total authors ids in chunk from twitter API: " + str(len(total_chunks)))
        i = 1
        for ids_in_chunk in ids_in_chunks:
            print("Chunk of authors ids: " + str(i) + "/" + str(len(total_chunks)))
            i += 1
            try:
                #num_of_get_users_requests = self._twitter_api_requester.get_num_of_get_users_requests()

                # if seconds_to_wait != 0:
                #     self.save_authors_and_connections_and_wait(users, author_type, insertion_type)
                #     users = []
                #     self._twitter_api_requester.init_num_of_get_users_requests()
                if self._num_of_get_twitter_users_requests < self._maximal_get_user_requests_in_window:

                    users = self.send_get_users_request_and_add_users(ids_in_chunk, are_user_ids,
                                                                      users)
                    total_users = list(set(total_users + users))
                    self._num_of_get_twitter_users_requests += 1


            except TwitterError as e:
                logging.info(e.message)
                sec = self._twitter_api_requester.get_sleep_time_for_get_users_request()
                logging.info("Seconds to wait from catched crush is: " + str(sec))
                count_down_time(sec)
                users = self.send_get_users_request_and_add_users(ids_in_chunk, are_user_ids, users)
                total_users = list(set(total_users + users))

            except Exception as e:
                logging.info(e.message)
                sec = self._twitter_api_requester.get_sleep_time_for_get_users_request()
                logging.info("Seconds to wait from catched crush is: " + str(sec))
                count_down_time(sec)
                users = self.send_get_users_request_and_add_users(ids_in_chunk, are_user_ids, users)
                total_users = list(set(total_users + users))

        print("--- Finishing handle_get_users_request --- ")
        logging.info("--- Finishing handle_get_users_request --- ")
        # self.save_authors_and_connections(users, author_type, insertion_type)
        return total_users

    def save_authors_and_connections_and_wait(self, total_twitter_users, author_type, inseration_type):
        self.save_authors_and_connections(total_twitter_users, author_type, inseration_type)

        seconds_to_wait = self._twitter_api_requester.get_sleep_time_for_get_users_request()
        self.count_down_time(seconds_to_wait)

    def save_authors_and_connections(self, total_twitter_users, author_type, inseration_type):
        self.convert_twitter_users_to_authors_and_save(total_twitter_users, author_type, inseration_type)

    def send_get_users_request_and_add_users(self, ids_in_chunk, are_user_ids, total_twitter_users):
        twitter_users = self.send_get_users_request(ids_in_chunk, are_user_ids)
        total_twitter_users = total_twitter_users + twitter_users
        return total_twitter_users

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
