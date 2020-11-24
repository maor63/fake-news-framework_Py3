from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class SnopesSpider(BaseSpider):
    name = "snopes_spider"
    scraper_url = 'https://www.snopes.com/fact-check/page/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(1120)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages + 1)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.find('div', class_='media-list').findAll('article')
        for element in contents:
            url = element.a['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            yield rec

    def parse_article(self, response):
        url = response.url
        element = response.meta['element']

        article_page = self.get_page_soup(response)
        title = element.h5.text
        description = element.find('p', class_='subtitle').text
        date_str = response.css('div.dates-wrapper li.date-item span.date-published::text').extract_first()
        verdict_date = datetime.strptime(date_str, '%d %B %Y')
        # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')
        article_page_all_categories = article_page.find('ol', class_='breadcrumb').find_all('li')
        category = article_page_all_categories[len(article_page_all_categories) - 1].a.text.strip()
        claim = article_page.find('div', class_='claim').text.strip()
        label = article_page.find('div', class_='media rating').find('h5').text.strip()
        # tags = ','.join(self.extract_tags(claim))
        tags = ''
        img_src = article_page.find('div', class_='image-wrapper').find('img')['data-lazy-src'].split('.jpg')[
                      0] + '.jpg'

        row_data = {'domain': self.name,
                    'title': title,
                    'claim': claim,
                    'description': description,
                    'url': url,
                    'verdict_date': verdict_date,
                    'tags': tags,
                    'category': category,
                    'label': label,
                    'image_src': img_src}
        yield self.export_row(**row_data)
