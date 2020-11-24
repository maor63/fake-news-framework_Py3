from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class TruthOrFictionSpider(BaseSpider):
    name = "truth_or_fiction_spider"
    scraper_url = 'https://www.truthorfiction.com/category/fact-checks/page/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(121)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.findAll('article')
        for element in contents:
            url = element.find('a')['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            rec.meta['page_soup'] = page_soup
            yield rec

    def parse_article(self, response):
        # open article page
        url = response.url
        element = response.meta['element']
        page_soup = response.meta['page_soup']
        article_page = self.get_page_soup(response)

        title = element.find('h2').text
        description = article_page.find('meta', attrs={'name': 'description'})['content']
        verdict_date = datetime.strptime(
            article_page.find('meta', attrs={'name': 'weibo:article:create_at'})['content'].split()[0], '%Y-%m-%d')
        # verdict_date = datetime.strftime(verdict_datetime, '%d/%m/%Y')
        category = 'Fact Checks'

        all_categories = page_soup.find('span', class_='cat-links').find_all('a')
        for categ in all_categories:
            if categ.text.strip() != 'Disinformation' and categ.text.strip() != 'Fact Checks':
                category = categ.text
                break
        label = article_page.find('div', class_='rating-description').text.strip()
        claim = article_page.find('div', class_='claim-description').text.strip()
        site_tags = []
        tags_content = article_page.find('ul', class_='tt-tags')
        if tags_content:
            tags_content = tags_content.find_all('li')
            for tag in tags_content:
                site_tags.append(tag.text.strip())
        tags = self.clean_site_tags(site_tags)
        tags = self.get_optional_tags(tags, self.extract_tags(claim))
        # img_src
        try:
            img_src = article_page.find('a', class_='tt-thumb')['href']
        except Exception:
            img_src = ''

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
