from commons.commons import *
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd


class GuidComputeUpdater(AbstractController):

    def __init__(self, db):
        super(GuidComputeUpdater, self).__init__(db)
        self._author_table_name = self._config_parser.eval(self.__class__.__name__, "author_table_name")
        self._posts_table_name = self._config_parser.eval(self.__class__.__name__, "posts_table_name")
        self._claims_table_name = self._config_parser.eval(self.__class__.__name__, "claims_table_name")
        self._claim_tweet_connection_table_name = self._config_parser.eval(self.__class__.__name__,
                                                                           "claim_tweet_connection_table_name")
        self._post_id_old_to_new_dict = {}
        self._claim_id_old_to_new_dict = {}

    def execute(self, window_start):
        print("Update authors table")
        authors_table = self._update_authors_table()
        print("Update posts table")
        posts_table = self._update_posts_table()
        print("Update claims table")
        claims_table = self._update_claims_table()
        print("Update claim_tweet_connection table")
        claim_tweet_connection_table = self._update_claim_tweet_connection_table()
        print("Save data to DB")
        claim_tweet_connection_table.to_sql(self._claim_tweet_connection_table_name, con=self._db.engine,
                                            if_exists='replace', index=False)
        claims_table.to_sql(self._claims_table_name, con=self._db.engine, if_exists='replace', index=False)
        posts_table.to_sql(self._posts_table_name, con=self._db.engine, if_exists='replace', index=False)
        authors_table.to_sql(self._author_table_name, con=self._db.engine, if_exists='replace', index=False)

    def _update_claim_tweet_connection_table(self):
        claim_tweet_connection_table = pd.read_sql_table(self._claim_tweet_connection_table_name, self._db.engine)
        claim_tweet_connection_table = claim_tweet_connection_table.apply(self.update_claim_id_post_id, axis=1)
        return claim_tweet_connection_table

    def _update_claims_table(self):
        claims_table = pd.read_sql_table(self._claims_table_name, self._db.engine)
        updated_claim_ids = claims_table.apply(self.calculate_claim_id, axis=1)
        old_claim_id_to_new_df = pd.DataFrame()
        old_claim_id_to_new_df['old'] = claims_table['claim_id']
        old_claim_id_to_new_df['new'] = updated_claim_ids
        claims_table['claim_id'] = updated_claim_ids
        self._claim_id_old_to_new_dict = dict(old_claim_id_to_new_df.to_records(index=False))
        return claims_table

    def _update_posts_table(self):
        posts_table = pd.read_sql_table(self._posts_table_name, self._db.engine)
        posts_table['created_at'] = posts_table['date'].apply(self._clean_created_at_field)
        posts_table['author_guid'] = posts_table['author'].apply(compute_author_guid_by_author_name)
        updated_post_ids = posts_table.apply(self.calculate_post_guid, axis=1)
        old_post_id_to_new_df = pd.DataFrame()
        old_post_id_to_new_df['old'] = posts_table['post_id']
        old_post_id_to_new_df['new'] = updated_post_ids
        posts_table['guid'] = posts_table['post_id'] = updated_post_ids
        self._post_id_old_to_new_dict = dict(old_post_id_to_new_df.to_records(index=False))
        return posts_table

    def _clean_created_at_field(self, record):
        if isinstance(record, str):
            post_date = record
        else:
            post_date = date_to_str(record)
        return post_date

    def _update_authors_table(self):
        authors_table = pd.read_sql_table(self._author_table_name, self._db.engine)
        authors_table['author_guid'] = authors_table['name'].apply(compute_author_guid_by_author_name)
        return authors_table

    def update_claim_id_post_id(self, record):
        record['claim_id'] = self._claim_id_old_to_new_dict[record['claim_id']]
        record['post_id'] = self._post_id_old_to_new_dict[record['post_id']]
        return record

    def calculate_claim_id(self, record):
        # title = record['title'] if record['title'] else ''
        url = record['url'] if record['url'] else self._social_network_url
        if isinstance(record['verdict_date'], str):
            verdict_date = record['verdict_date']
        else:
            verdict_date = date_to_str(record['verdict_date'])
        return compute_post_guid(self._social_network_url, url.strip(), verdict_date)

    def calculate_post_guid(self, record):
        if isinstance(record['date'], str):
            created_at = record['date']
        else:
            created_at = date_to_str(record['date'])
        post_url = record['url'] if record['url'] else self._social_network_url
        author_name = record['author']

        return compute_post_guid(post_url, author_name, created_at)
