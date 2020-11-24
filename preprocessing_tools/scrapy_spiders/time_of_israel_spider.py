from datetime import datetime
from urllib.request import urlopen
import parse
import scrapy
from bs4 import BeautifulSoup

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class TheTimeOfIsraelSpider(BaseSpider):
    name = "the_time_of_israel_spider"
    scraper_url = 'https://www.timesofisrael.com/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        categories = ['israel-and-the-region', 'jewish-times', 'start-up-israel', 'israel-inside']
        for category in categories:
            category_url = '{}{}/'.format(self.scraper_url, category)
            total_pages = self.get_category_max_page(category_url)
            urls = list(['{}page/{}'.format(category_url, i) for i in range(1, total_pages + 1)])
            for url in urls:
                request = scrapy.Request(url=url, callback=self.parse)
                request.meta['category'] = category
                yield request

    def get_category_max_page(self, category_url):
        category_2_page_url = '{}page/2'.format(category_url)
        soup = BeautifulSoup(urlopen(category_2_page_url), "html.parser")
        title = soup.find('title').text
        max_page_container = title.split('|')[1].strip()
        max_page = parse.parse('Page 2 of {}', max_page_container)[0]
        return self.get_pages_to_crawl(int(max_page))

    def parse(self, response):
        assert isinstance(response, scrapy.http.Response)
        articles_url = response.css('section div.item.news div.media a::attr(href)').extract()
        for article_url in articles_url:
            request = scrapy.Request(url=article_url, callback=self.parse_article)
            request.meta['category'] = response.meta['category']
            yield request

    def parse_article(self, response):
        assert isinstance(response, scrapy.http.Response)
        url = response.url
        title = response.css('header h1.headline::text').extract_first().replace('"', '')
        claim = response.css('header h2.underline::text').extract_first().replace('"', '')
        paragraphs = list(map(str.strip, response.css('div.article-content div.the-content p::text').extract()))
        description = ' '.join(paragraphs).replace('"', '')
        verdict_date_str = response.css('div.under-headline span.date::text').extract_first()
        if 'Today' in verdict_date_str:
            verdict_date_str = verdict_date_str.replace('Today', datetime.today().strftime('%d %B %Y'))
        if 'jewishnews' in url:
            date_format = '%B %d, %Y, %I:%M %p'
        else:
            date_format = '%d %B %Y, %I:%M %p'
        verdict_date = datetime.strptime(verdict_date_str, date_format)
        tags = ','.join(response.css('div.article-topics ul li a::text').extract())
        img_src = response.css('div.media a::attr(href)').extract_first()
        row_data = {'domain': self.name,
                    'title': title,
                    'claim': claim,
                    'description': description,
                    'url': url,
                    'verdict_date': verdict_date,
                    'tags': tags,
                    'category': response.meta['category'],
                    'label': 'TRUE',
                    'image_src': img_src}
        yield self.export_row(**row_data)
