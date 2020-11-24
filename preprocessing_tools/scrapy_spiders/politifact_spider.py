from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class PolitifactSpider(BaseSpider):
    name = "politifact_spider"
    scraper_url = 'https://www.politifact.com/truth-o-meter/statements/?page='
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(820)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.find('section', class_='scoretable').findAll('div', class_='scoretable__item')
        for element in contents:
            url = self.scraper_url.split('.com')[0] + '.com' + element.find('a', class_='link')['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            yield rec

    def parse_article(self, response):
        url = response.url
        element = response.meta['element']

        article_page = self.get_page_soup(response)
        title = article_page.find('h1', class_='article__title').text
        try:
            description = article_page.find("meta", property="og:description")['content'].strip()
            verdict_date_full = element.find('span', class_='article__meta').text.strip().split(',')
            verdict_date = verdict_date_full[1].strip().split(' ')[0] + ' ' + self.replace_suffix_in_date(
                verdict_date_full[1].strip().split(' ')[1]) + ', ' + verdict_date_full[2].strip()
            verdict_date = datetime.strptime(verdict_date, '%B %d, %Y')
            # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')

            category = 'Politics'
            claim = element.find('div', class_='statement__source').text.strip() + '-' + element.find('a',
                                                                                                      class_='link').text.strip()
            label = element.find('div', class_='meter').find('img')['alt'].strip()
            tags = ','.join(self.extract_tags(claim))
            img_src = element.find('div', class_='statement__body').find('img')['src']

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
        except:
            yield None
