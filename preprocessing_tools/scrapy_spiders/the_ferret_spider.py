from datetime import datetime
import scrapy
from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class TheFerretSpider(BaseSpider):
    name = "the_ferret_spider"
    scraper_url = 'https://theferret.scot/fact-check/page/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(12)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages + 1)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        assert isinstance(response, scrapy.http.Response)
        claims_urls = response.css('div[id="posts"] article h1 a::attr(href)').extract()
        tags_list = response.css('header.entry-header h2.entry-subtitle')
        for url, tags in zip(claims_urls, tags_list):
            request = scrapy.Request(url, callback=self.parse_article)
            request.meta['tags'] = ','.join(tags.css('a::text').extract())
            yield request

    def parse_article(self, response):
        assert isinstance(response, scrapy.http.Response)
        url = response.url
        title = response.css('header.cover-header h1::text').extract_first()
        description = response.css('div.entry-content.aesop-entry-content strong:first_of_type::text').extract_first()
        claim = ' '.join(response.css('div.entry-content.aesop-entry-content p::text').extract())
        publish_str = response.css('span.posted-on time.published::text').extract_first()
        verdict_date = datetime.strptime(publish_str, '%B %d, %Y')
        # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')
        has_verdict = lambda text: text.startswith('Ferret Fact Service verdict')
        label = list(filter(has_verdict, response.css('div.entry-content.aesop-entry-content h3::text').extract()))[0]
        label = label.split(':')[-1].strip()
        tags = response.meta['tags']

        row_data = {'domain': self.name,
                    'title': title,
                    'claim': claim,
                    'description': description,
                    'url': url,
                    'verdict_date': verdict_date,
                    'tags': tags,
                    'category': 'ferret',
                    'label': label,
                    'image_src': ''}
        yield self.export_row(**row_data)
