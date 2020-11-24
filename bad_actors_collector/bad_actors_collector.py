#
# Created by Aviad on 28-May-16 7:31 PM.
#


from commons.commons import *
from commons.consts import *
from commons.method_executor import Method_Executor
from preprocessing_tools.abstract_controller import AbstractController
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api


class BadActorsCollector(Method_Executor):

    def __init__(self, db):
        AbstractController.__init__(self, db)

        self._actions = self._config_parser.eval(self.__class__.__name__, "actions")

        self._targeted_twitter_author_ids = self._config_parser.eval(self.__class__.__name__, "targeted_twitter_author_ids")

        self._targeted_twitter_post_ids = self._config_parser.eval(self.__class__.__name__, "targeted_twitter_post_ids")

        self._targeted_twitter_author_names = self._config_parser.eval(self.__class__.__name__, "targeted_twitter_author_names")

        self._social_network_crawler = Twitter_Rest_Api(db)

    def setUp(self):
        pass

    def crawl_bad_actors_followers(self):
        print("---crawl_bad_actors_followers_and_retweeters ---")
        bad_actor_type = Author_Type.BAD_ACTOR
        bad_actors_collector_inseration_type = DB_Insertion_Type.BAD_ACTORS_COLLECTOR
        connection_type = Author_Connection_Type.FOLLOWER
        are_user_ids = True
        self._social_network_crawler.crawl_users_by_author_ids(self._targeted_twitter_author_ids,connection_type,
                                                               bad_actor_type,
                                                               are_user_ids,
                                                               bad_actors_collector_inseration_type)


        self._db.convert_temp_author_connections_to_author_connections(self._domain)


    def crawl_bad_actors_retweeters(self):
        bad_actor_type = Author_Type.BAD_ACTOR
        bad_actors_collector_inseration_type = DB_Insertion_Type.BAD_ACTORS_COLLECTOR
        are_user_ids = True
        self._social_network_crawler.crawl_retweeters_by_post_id(self._targeted_twitter_post_ids, are_user_ids, bad_actor_type,
                                                                 bad_actors_collector_inseration_type)

    def mark_missing_bad_retweeters_retrieved_from_vico(self):
        print("mark_missing_bad_retweeters_retrieved_from_vico")
        missing_bad_actors = []
        i = 0

        cursor = self._db.get_cooperated_authors(self._targeted_twitter_author_names, self._domain)

        targeted_twitter_author_guid_generator = self._db.result_iter(cursor)

        for missing_author_guid in targeted_twitter_author_guid_generator:
            i += 1
            missing_author_guid = str(missing_author_guid[0])
            result = self._db.get_author_by_author_guid_and_domain(missing_author_guid, self._domain)
            if len(result) > 0:
                missing_author = result[0]

                missing_author.author_type = Author_Type.BAD_ACTOR
                missing_author.mark_missing_bad_actor_retweeters_insertion_date = self._window_start

                missing_bad_actors.append(missing_author)

            else:
                logging.info("GUID = " + missing_author_guid)
        logging.info("number of missing bad actors found are:" +  str(len(missing_bad_actors)))
        self._db.add_authors(missing_bad_actors)


