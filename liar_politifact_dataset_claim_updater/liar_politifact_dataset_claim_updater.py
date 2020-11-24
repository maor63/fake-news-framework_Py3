

from commons.commons import str_to_date
from commons.method_executor import Method_Executor
import pandas as pd
import numpy as np
import re

from preprocessing_tools.politi_fact_posts_crawler.politi_fact_posts_crawler import PolitiFactPostsCrawler

__author__= "Aviad Elyashar"

#########################################################################################################
# This module is responsible for updating the claims by a click instead of updating the records manually#
#########################################################################################################

class LiarPolitifactDatasetClaimUpdater(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._source_file_full_path = self._config_parser.eval(self.__class__.__name__, "source_file_full_path")

        self._politifact_crawler = PolitiFactPostsCrawler(db)
    def update_records(self):
        df_to_update = pd.read_csv(self._source_file_full_path)
        total_claims_to_update = df_to_update.shape[0]
        self._post_id_post_dict = self._db.get_post_dictionary()

        posts_to_update = []
        i = 0
        for index, row in df_to_update.iterrows():
            updated_post = self._update_post_by_row(row)

            i += 1
            msg = "\r Processing claims {0}/{1}".format(i, total_claims_to_update)
            print(msg, end="")

            posts_to_update.append(updated_post)

        self._db.addPosts(posts_to_update)

    def _update_post_by_row(self, row):
        post_id = str(row['post_id'])
        author_name = row['author']
        content = str(row['current_title'])
        description = row['previous_title']
        description = str(description) if description is not np.nan else description
        url = str(row['url'])
        publication_date_str = row['publication_date']
        publication_date = str_to_date(publication_date_str)
        num_of_returned_tweets = row['num_of_returned_tweets']
        keywords = str(row['keywords'])
        post_type = str(row['post_type'])

        post = self._post_id_post_dict[post_id]
        post.content = content

        if description is not np.nan:
            post.description = description

        post.url = url
        post.date = publication_date
        post.tags = keywords

        return post

    def fill_data_from_politifact(self):
        posts_to_update = []
        optional_posts_to_update = self._db.get_posts_with_no_dates()
        num_of_optional_posts_to_update = len(optional_posts_to_update)

        post_id_post_dict = self._db.get_post_dictionary()
        i = 0
        for post_to_update in optional_posts_to_update:
            i += 1
            msg = "\r Processing posts: [{0}/{1}]".format(i, num_of_optional_posts_to_update)
            print(msg, end="")

            post_id = post_to_update.post_id
            try:
                retrieved_posts = self._politifact_crawler.get_claim_by_id(post_id)
                if len(retrieved_posts) > 0:
                    retrieved_post_dict = retrieved_posts[0]
                    post = post_id_post_dict[post_id]

                    self._set_ruling_date(retrieved_post_dict, post)
                    self._set_url(retrieved_post_dict, post)

                    posts_to_update.append(post)

            except Exception as e:
                print(e)

        self._db.addPosts(posts_to_update)

    def _set_ruling_date(self, retrieved_post_dict, post):
        ruling_date = retrieved_post_dict['ruling_date']
        pattern = "^([^\.]+)T([^\.]+)$"
        match = re.match(pattern, ruling_date)
        group_tuple = match.groups()

        date = group_tuple[0]
        time = group_tuple[1]
        ruling_date_str = date + " " + time
        ruling_date = str_to_date(ruling_date_str)

        post.date = ruling_date

    def _set_url(self, retrieved_post_dict, post):
        url = retrieved_post_dict['statement_url']
        post.url = str("http://www.politifact.com" + url)

