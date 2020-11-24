from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class FactscanSpider(BaseSpider):
    name = "factscan_spider"
    scraper_url = 'http://factscan.ca/page/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(27)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.findAll('article')
        for element in contents:
            url = element.find('h1').find('a')['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            yield rec

    def parse_article(self, response):
        url = response.url
        element = response.meta['element']
        article_page = self.get_page_soup(response)
        # title
        title = element.find('h1').find('a')['title']
        description = article_page.find('meta', attrs={'property': 'og:description'})['content']
        verdict_date = datetime.strptime(article_page.find('time').text, '%B %d, %Y')
        # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')
        label = article_page.find('div', class_='fact-check-icon').find('img')['alt'].split(':')[1].strip()
        site_tags = []
        for tag in article_page.find('span', class_='post-category').findAll('a'):
            site_tags.append(tag.text.strip())
        tags = self.clean_site_tags(site_tags)
        tags = self.get_optional_tags(tags, self.extract_tags(title))
        img_src = article_page.find('div', class_='post-content').findAll('img')[1]['src']

        row_data = {'domain': self.name,
                    'title': title,
                    'claim': title,
                    'description': description,
                    'url': url,
                    'verdict_date': verdict_date,
                    'tags': tags,
                    'category': 'Politics',
                    'label': label,
                    'image_src': img_src}
        yield self.export_row(**row_data)
