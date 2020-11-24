from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class GossipCopSpider(BaseSpider):
    name = "gossip_cop_spider"
    scraper_url = 'https://www.gossipcop.com/page/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(367)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.find('div', {'id': 'posts'}).findAll('div', class_='post')
        for element in contents:
            url = element.find('h2').find('a')['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            yield rec

    def parse_article(self, response):
        url = response.url
        element = response.meta['element']

        article_page = self.get_page_soup(response)
        title = element.find('h2').find('a').text.strip()
        description = article_page.find('meta', attrs={'name': 'description'})['content']
        verdict_date_full = article_page.find('span', class_='dateline').text.split(',')

        verdict_date = verdict_date_full[1].split()[0].strip() + ' ' + verdict_date_full[1].split()[1].strip() + ', ' + \
                       verdict_date_full[2].strip()
        verdict_date = datetime.strptime(verdict_date, '%B %d, %Y')
        # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')
        category = 'Entertainment'
        claim = title.split('?')[0].strip()
        try:
            label = article_page.select('div[class^=meter]')[0].find('span').text.split(':')[1].strip()
        except:
            label = None
        site_tags = []
        for tag in article_page.find('p', class_='tags').findAll('a'):
            site_tags.append(tag.text.strip())
        tags = self.clean_site_tags(site_tags)
        tags = self.get_optional_tags(tags, self.extract_tags(claim))
        img_src = element.find('img')['src']

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
