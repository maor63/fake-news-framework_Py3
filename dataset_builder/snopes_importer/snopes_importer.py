import csv
import datetime
import uuid

import scrapy
import timestring
from scrapy import cmdline
from DB.schema_definition import Claim,DB
from preprocessing_tools.abstract_controller import AbstractController


class ScrapySnopsCrawler(scrapy.Spider):


    name = "snops_spider"

    start_urls=['https://www.snopes.com/fact-check','https://www.snopes.com/ap']
    # start_urls=['https://www.snopes.com/ap']


    def parse(self, response):
        LINK_SELECTOR = '.list-group-item a ::attr(href)'
        for href in response.css(LINK_SELECTOR):
            url=response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_dir_contents)
        next_url = response.css('.btn-next ::attr(href)').extract_first()
        yield scrapy.Request(next_url,callback=self.parse)

    def parse_dir_contents(self, response):
        TITLE_SELECTOR = '.card-title ::text'
        CATERGORY_SELECTOR= '.breadcrumb-item a ::text'
        CLAIM_SELECTOR = '.claim ::text'
        r = response.css(CATERGORY_SELECTOR)
        RATING_SELECTOR='.rating-name ::text'
        DATE_SELECTOR = '.date-published ::text'
        yield {
            'title': response.css(TITLE_SELECTOR).extract_first(),
            'claim': response.css(CLAIM_SELECTOR).extract_first(),
            'main-category': response.css(CATERGORY_SELECTOR)[0].extract(),
            'sub-category': response.css(CATERGORY_SELECTOR)[1].extract(),
            'rating': response.css(RATING_SELECTOR).extract_first(),
            'date': response.css(DATE_SELECTOR).extract_first(),
            'url':response.request.url,
        }
    def run_importer(self,file_path):
        query = 'scrapy runspider dataset_builder/snopes_importer/snopes_importer.py -o {} -t csv'.format(file_path)
        cmdline.execute(query.split())

class ScrapySnopes(AbstractController):
    def __init__(self, db):
        super(ScrapySnopes, self).__init__(db)
        self.output_file = self._config_parser.eval(self.__class__.__name__,'output_path')
        self.to_scrape = self._config_parser.eval(self.__class__.__name__,'to_scrape')

    def setUp(self):
        pass

    def is_well_defined(self):
        return True
    def execute(self, window_start):
        s= ScrapySnopsCrawler()
        if (self.to_scrape):
            s.run_importer(self.output_file)
        self.import_to_db(self.output_file)

    def import_to_db(self, file_path):
        with open(file_path, 'rb') as f:
            data = csv.reader(f)
            res = []
            for row in data:
                try:
                    c = Claim()
                    c.title = str(row[3])
                    c.claim_id = str(uuid.uuid4().__str__())
                    c.description = str(row[0])
                    c.category = str(row[4])
                    c.verdict = str(row[1])
                    c.domain= "Claim"
                    c.verdict_date = datetime.datetime.strptime(row[6],'%d %B %Y')
                    c.url = str(row[2])
                except Exception as e:
                    continue
                res.append(c)
                print(','.join(row))
            self._db.add_claims(res)
