# encoding: utf-8
# need to be added to the system


from DB.schema_definition import Post, date, Claim
from commons.commons import compute_post_guid, compute_author_guid_by_author_name
from commons.method_executor import Method_Executor
import pandas as pd
import re

__author__ = "Aviad Elyashar"


class FakeNewsSnopesImporter(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_csv_file = self._config_parser.eval(self.__class__.__name__, "input_csv_file")

        # There is no author so the website would be the author. We should not include this author in the analysis.
        self._author_name = str("snopes")

    # def execute(self, window_start=None):
    #    self.read_file_and_create_claims()

    def read_file_and_create_claims(self):
        snopes_csv_df = pd.read_csv(self._input_csv_file)
        num_of_records = snopes_csv_df.shape[0]

        claims = []
        i = 0
        for index, row in snopes_csv_df.iterrows():
            i += 1
            print("\r Convert row to claim {0}:{1}".format(i, num_of_records), end="")
            claim = self._convert_row_to_claim(row)
            claims.append(claim)

        self._db.addPosts(claims)

    def _convert_row_to_post(self, row):
        post = Post()

        claim_id = str(row['claim_id'])
        title = str(row['title'], errors='replace')
        post.content = title

        description = str(row['description'], errors='replace')
        post.description = description

        url = str(row['url'])
        post.url = url

        publication_date = row['publication_date']
        post.date = date(publication_date)

        post_guid = compute_post_guid(self._social_network_url, claim_id, publication_date)
        post.guid = post_guid
        post.post_id = post_guid
        post.domain = self._domain
        post.author = self._author_name
        author_guid = compute_author_guid_by_author_name(self._author_name)
        post.author_guid = author_guid
        post.post_osn_guid = post_guid

        keywords = str(row['keywords'])
        post.tags = keywords

        post_type = str(row['post_type'])
        post.post_type = post_type

        return post

    def _convert_row_to_claim(self, row):
        claim = Claim()

        # claim_id = unicode(row['claim_id'])
        title = str(row['title'], errors='replace')
        claim.title = title

        description = str(row['description'], errors='replace')
        claim.description = description

        url = str(row['url'])
        claim.url = url

        verdict_date = row['verdict_date']
        claim.verdict_date = date(verdict_date)

        post_guid = compute_post_guid(self._social_network_url, url, verdict_date)
        claim.claim_id = post_guid

        claim.domain = str(row.get('keywords', self._domain))

        keywords = str(row['keywords'])
        claim.keywords = keywords

        verdict = str(row['verdict'])
        claim.verdict = verdict

        claim.category = str(row['main_category'])
        claim.sub_category = str(row['secondary_category'])

        return claim

    def convert_comma_to_or_in_keywords_claims(self):
        claims = self._db.get_claims()
        for claim in claims:
            keywords_str = claim.keywords
            new_keywords_str = re.sub(",", "||", keywords_str)
            claim.keywords = new_keywords_str
        self._db.addPosts(claims)
