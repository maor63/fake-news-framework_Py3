import json
import requests
from scrapy.selector import Selector


class ArticleNewYorkTimes():
    def __init__(self, title, body,authors,date):
        self.title=title
        self.body = body
        self.authors=authors
        self.date=date

    def is_null(self):
        return self.title is None

    def __str__(self):
        return self.title+' '+self.date

def parse_page_new_york_times(url):
    txt=requests.get(url).text
    page=Selector(text=txt)
    title= page.xpath('//header/h1/span/text()').extract_first()
    body_list=page.xpath('//*[contains(@name,"articleBody")]/div/div/p/text()').extract()
    body=' '.join(body_list)
    authors=page.xpath('//*[contains(@itemprop,"author creator")]/a/span/text()').extract()
    date = page.xpath('//header/div/div/ul/li/time/text()').extract_first()

    return ArticleNewYorkTimes(title,body,authors,date)


if __name__ == '__main__':
    api_key='a4d67fc07cab4a26b07a4fa80dea7711'
    source_name='the-new-york-times'
    url = 'https://newsapi.org/v2/everything?sources={}&apiKey={}'.format(source_name,api_key)
    request= requests.get(url).text
    page_json= json.loads(request)

    parsed = []
    for article in page_json['articles']:
        url = article['url']
        p=parse_page_new_york_times(url)
        if not p.is_null():
            parsed.append(p)
    #TODO do somethign with the parsed items
    #TODO embed to system
    pass
