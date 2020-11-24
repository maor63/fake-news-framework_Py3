import logging

class KeyAuthorsModel:
    '''
    This module is in charge of calculating the SumTFIDF and MaxTFIDF scores for each author.
    This is done by creating the relevant views and tables.
    The operation of this module depends on the creation of the topics and post_topic_mapping tables created by the
    autotopic_executor module
    '''
    def __init__(self, db):
        self._db = db


    def setUp(self):
        self._db.delete_post_representativeness_data()

    def execute(self, window_start=None):
        if self.db_contains_required_tables():
            self.load_views_and_tables()
        else:
            logging.error("Cannot run KeyAuthorsModel as the post_topic_mapping does not appear in the db")

    def db_contains_required_tables(self):
        '''
        the views and tables of the KeyAuthorsModel depend on the post_topic_mapping table.
        If it doesn't exist, the module cannot operate as required.
        '''
        return self._db.is_post_topic_mapping_table_exist()

    def load_views_and_tables(self):
        '''
        Loads the views and tables that rely on topic extraction
        and are required for the  SumTFIDF and MaxTFIDF calculation
        '''
        self._db.create_uf_view()
        self._db.create_tf_view()
        self._db.create_total_url_frequency_view()
        self._db.create_topic_stats_view()
        self._db.create_tfidf_view()
        self._db.load_posts_representativeness_table()
        self._db.create_key_posts_view()
        self._db.create_key_authors_view()
