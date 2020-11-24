# Aviad Elyashar

import logging

from Twitter_API.twitter_api_requester import TwitterApiRequester
from dataset_builder.graph_builder import GraphBuilder
import time

class GraphBuilder_Friends(GraphBuilder):
    def __init__(self, db):
        GraphBuilder.__init__(self, db)
        self._twitter_api_requester = TwitterApiRequester(1)

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting authors from DB ")

        authors = self._db.get_authors()
        type_connections = self._db.get_author_connections_by_type('friend')
        authors_with_connections = set(con[0] for con in type_connections)
        authors = [a for a in authors if a.author_guid not in authors_with_connections]

        author_osn_id_author_guid_dict = self._create_author_osn_id_author_guid_dictionary(authors)
        author_osn_ids = set(author_osn_id_author_guid_dict.keys())
        author_connections = []

        for i, author in enumerate(authors):
            author_osn_id = int(author.author_osn_id)
            print('\r get friends for author {}/{}'.format(str(i + 1), len(authors)), end='')
            try:
                friend_ids = self._twitter_api_requester.get_friend_ids_by_user_id(author_osn_id)
                friend_ids = set(friend_ids)
                mutual_friend_ids = friend_ids.intersection(author_osn_ids)
                if len(mutual_friend_ids) > 0:
                    mutual_friend_ids = list(mutual_friend_ids)
                    for mutual_friend_id in mutual_friend_ids:
                        author_guid = author.author_guid
                        mutual_friend_guid = author_osn_id_author_guid_dict[mutual_friend_id]
                        author_connection = self._db.create_author_connection(author_guid, mutual_friend_guid, 1.0,
                                                                          self._connection_type, self._window_start)

                        author_connections.append(author_connection)

                        if len(author_connections) == self._max_objects_without_saving:
                            self._db.add_author_connections_fast(author_connections)
                            author_connections = []
            except Exception as e:
                print('error for {}'.format(author_osn_id))
                print(e)
        self._db.add_author_connections_fast(author_connections)

    def _create_author_osn_id_author_guid_dictionary(self, authors):
        author_osn_id_author_guid_dict = {}
        for author in authors:
            if author.author_osn_id:
                author_osn_id = int(author.author_osn_id)
                author_guid = author.author_guid

                if author_osn_id not in author_osn_id_author_guid_dict:
                    author_osn_id_author_guid_dict[author_osn_id] = author_guid
        return author_osn_id_author_guid_dict






