from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class PolygraphSpider(BaseSpider):
    name = "polygraph_spider"
    scraper_url = 'https://www.polygraph.info/z/20382?p='
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(20)
        base_apis = ['20375', '20382', '20446', '20379', '20380']
        base_urls = ['https://www.polygraph.info/z/{}?p='.format(api) for api in base_apis]
        urls = []
        for base_url in base_urls:
            urls += list(['{}{}'.format(base_url, i) for i in range(1, total_pages)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.findAll('div', 'fc__hdr')
        for element in contents:
            url = self.scraper_url.split('/z/20382?p=')[0] + element.find('a', 'title')['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            yield rec

    def parse_article(self, response):
        url = response.url
        element = response.meta['element']

        article_page = self.get_page_soup(response)
        title = element.find('a', class_='title').find('h4').text
        description = article_page.find('meta', attrs={'name': 'description'})['content']
        verdict_datetime = article_page.find('span', class_='date').text.strip()
        if 'Last Updated:' in verdict_datetime:
            verdict_datetime = verdict_datetime.split('Last Updated:')[1].strip()
        verdict_date = datetime.strptime(verdict_datetime, '%B %d, %Y')
        # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')
        category = article_page.find('div', class_='category').text.strip()
        claim = article_page.find('div', class_='wsw').find('p').text.strip()
        label = article_page.find('div', class_='verdict-head').find('span', class_='').text.strip()
        site_tags = []
        tags_content = article_page.find('meta', attrs={'name': 'news_keywords'})['content'].split(',')
        for tag in tags_content:
            site_tags.append(tag.strip())
        tags = self.clean_site_tags(site_tags)
        tags = self.get_optional_tags(tags, self.extract_tags(claim))
        img_src = article_page.find('div', class_='img-wrap').find('img')['src']

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
