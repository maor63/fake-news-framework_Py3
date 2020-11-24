from datetime import datetime

import scrapy

from preprocessing_tools.scrapy_spiders.base_spider import BaseSpider


class ClimateFeedbackSpider(BaseSpider):
    name = "climate_feedback_spider"
    scraper_url = 'https://climatefeedback.org/claim-reviews/'
    custom_settings = BaseSpider.get_settings(name, 'output/')

    def start_requests(self):
        '''
        Initial url for crawling
        :return: [urls]
        '''
        total_pages = self.get_pages_to_crawl(8)
        urls = list(['{}{}'.format(self.scraper_url, i) for i in range(1, total_pages)])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_soup = self.get_page_soup(response)
        contents = page_soup.findAll('div', class_='row')[1:-1]
        for element in contents:
            url = element.find('a')['href']
            rec = scrapy.Request(url, callback=self.parse_article)
            rec.meta['element'] = element
            yield rec

    def parse_article(self, response):
        url = response.url
        element = response.meta['element']
        article_page = self.get_page_soup(response)
        # title
        title = element.find('a').text.strip()
        description = article_page.find('meta', attrs={'property': 'og:description'})['content']
        publish_on_tag = article_page.find('p', class_='small').find_next('p').text.strip().split('Published on:')[1]
        verdict_date_full = publish_on_tag.split('|')[0].strip().split(' ')

        day, month, year = [x.strip() for x in verdict_date_full][:3]
        verdict_date_str = '{} {}, {}'.format(month, day, year)
        verdict_date = datetime.strptime(verdict_date_str, '%b %d, %Y')
        claim = element.find('div', class_='feedpages-excerpt').text.strip()
        label = element.find('img', class_='fact-check-card__row__verdict__img')['src'].split('HTag_')[1].rstrip('.png')
        tags = ','.join(self.extract_tags(claim))
        image_tag = element.find('img', class_='feedpages__claim__container__illustration__screenshot__img')['src']
        img_src = image_tag.rstrip('.png')

        row_data = {'domain': self.name,
                    'title': title,
                    'claim': claim,
                    'description': description,
                    'url': url,
                    'verdict_date': verdict_date,
                    'tags': tags,
                    'category': 'Climate',
                    'label': label,
                    'image_src': img_src}
        yield self.export_row(**row_data)
