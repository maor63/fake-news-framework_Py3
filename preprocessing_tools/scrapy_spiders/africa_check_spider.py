from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class AfricaCheckSpider(BaseSpider):
    name = "africa_check_spider"
    scraper_url = 'https://africacheck.org/latest-reports/page/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(53)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.find('div', class_='col-sm-8 clearfix').findAll('article')
        for element in contents:
            url = element.find('h2').find('a')['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            yield rec

    def parse_article(self, response):
        url = response.url
        element = response.meta['element']
        article_page = self.get_page_soup(response)
        if article_page:
            yield self.extract_article_data(article_page, element, url)

    def extract_article_data(self, article_page, element, url):
        title = element.find('h2').text.strip()
        description = article_page.find('meta', attrs={'name': 'description'})['content']
        verdict_date_full = element.find('p', class_='date-published').text.strip().split('| ')[1].split(' ')
        verdict_date = verdict_date_full[1].strip() + ' ' + self.replace_suffix_in_date(
            verdict_date_full[0].strip()) + ', ' + verdict_date_full[2].strip()
        verdict_date = datetime.strptime(verdict_date, '%B %d, %Y')
        # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')
        try:
            category = element.find('ul', class_='tag-list').find('li').text.strip()
        except:
            category = None
        try:
            claim = article_page.find('div', class_='report-claim').find('p').text.strip()
            label = element.find('div', class_='verdict-stamp').text.strip()
            site_tags = []
            for tag in element.find('ul', class_='tag-list').findAll('li')[1:]:
                site_tags.append(tag.find('a').text.strip())
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
            return self.export_row(**row_data)
        except:
            return None
