import os
import shutil

import pandas as pd
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from DB.schema_definition import Post, Claim
from commons.commons import *
from preprocessing_tools.abstract_controller import AbstractController
from preprocessing_tools.scrapy_spiders.snopes_spider import *
from preprocessing_tools.scrapy_spiders.africa_check_spider import *
from preprocessing_tools.scrapy_spiders.climate_feedback_spider import *
from preprocessing_tools.scrapy_spiders.factscan_spider import *
from preprocessing_tools.scrapy_spiders.politifact_spider import *
from preprocessing_tools.scrapy_spiders.polygraph_spider import *
from preprocessing_tools.scrapy_spiders.gossipcop_spider import *
from preprocessing_tools.scrapy_spiders.truth_or_fiction_spider import *
from preprocessing_tools.scrapy_spiders.the_ferret_spider import *
from preprocessing_tools.scrapy_spiders.time_of_israel_spider import *
import sys


class SpiderManager(AbstractController):

    def __init__(self, db):
        super(SpiderManager, self).__init__(db)
        crawlers_names = self._config_parser.eval(self.__class__.__name__, "crawlers_names")
        self._claims_output_path = self._config_parser.eval(self.__class__.__name__, "claims_output_path")
        self._crawl_top_claims = self._config_parser.eval(self.__class__.__name__, "crawl_top_claims")
        self.crawlers = [getattr(sys.modules[__name__], crawler_name) for crawler_name in crawlers_names]

    def setUp(self):
        if os.path.isdir(self._claims_output_path):
            shutil.rmtree(self._claims_output_path)

    def execute(self, window_start=None):
        self.crawl(claims_csv_output_path=self._claims_output_path, only_new_claims=self._crawl_top_claims)
        self.add_claims_to_db(self._claims_output_path)

    def crawl(self, claims_csv_output_path, only_new_claims):
        process = CrawlerProcess(settings={
            'UPDATE': only_new_claims,
            'FEED_FORMAT': 'csv',
        })
        for crawler in self.crawlers:
            crawler.set_output_path(claims_csv_output_path)
            process.crawl(crawler)
        process.start()
        process.stop()

    def add_claims_to_db(self, output_path):
        for claims_csv in os.listdir(output_path):
            csv_path = os.path.join(output_path, claims_csv)
            claims = self.get_claims(csv_path)
            self._db.addPosts(claims)

    def get_claims(self, csv_path):
        claims = []
        claims_df = pd.read_csv(csv_path)
        for record in claims_df.to_records(index=False):
            claims.append(self.create_claim_from_record(record))
        return claims

    def create_claim_from_record(self, series):
        claim = Claim()
        claim.title = str(series['title'])
        claim.description = str(series['description'])
        claim.url = str(series['url'])
        claim.verdict_date = str_to_date(series['verdict_date'])
        claim.verdict = str(series['verdict'])
        claim.category = str(series['main_category'])
        claim.keywords = str(series['keywords'])
        claim.domain = str(series['domain'])

        post_guid = compute_post_guid(self._social_network_url, claim.url, date_to_str(claim.verdict_date))
        claim.claim_id = post_guid
        return claim


