import types
import unittest
from unittest import TestCase

from DB.schema_definition import DB
from preprocessing_tools.instagram_crawler.instagram_crawler import InstagramCrawler


class StubMockup(object):
    def __new__(cls, **attributes):
        result = object.__new__(cls)
        result.__dict__ = attributes
        return result


def get_file_text(filename):
    with open(filename, 'r') as fils:
        x = fils.read()
    return x


f = lambda self, url: jsons.setdefault(url, StubMockup(text=get_file_text('data/input/instagramCrawler/no_comments.json')))
jsons = {
    'https://www.instagram.com/graphql/query/?query_hash=42323d64886122307be10013ad2dcc44&variables={"id":"491527077","first":50,"after":null}':
        StubMockup(text=get_file_text('data/input/instagramCrawler/variables={"id"-"491527077","first"-50,"after"-null}.json')),
    'https://www.instagram.com/graphql/query/?query_hash=42323d64886122307be10013ad2dcc44&variables={"id":"491527077","first":50,"after":"QVFCam10VUF2c1hoY1RWMnhsd3J3TVVFRDNTVXVudkphMWlCZXV0SFEySnJmNU1ZdjVoaERJRzd0TmpCTzNmR1VQMlk1ZC0yZ2lRSnV3MENDdFJPTENzdg=="}':
        StubMockup(text=get_file_text('data/input/instagramCrawler/variables={"id"-"491527077","first"-50,"after"-"QVFDS3RBT18xeWFtc01kcTlOUE1TMFNiakJ3R2hNMnU4OHZLel9yWWpYUndySkdBeVppU0VoN1VGTEtmZ0lxRUx4SkthN2NieU5VeThyNlVYSkFvWm9HaA=="}.json')),
    'https://www.instagram.com/graphql/query/?query_hash=33ba35852cb50da46f5b5e889df7d159&variables={"shortcode":"BxWnbUcA3Ov","first"-50,"after":null}':
        StubMockup(text=get_file_text('data/input/instagramCrawler/variables={"shortcode"-"BxWnbUcA3Ov","first"-50,"after"-null}.json')),
    'https://www.instagram.com/manchesterunited/?__a=1':
        StubMockup(text=get_file_text('data/input/instagramCrawler/manchesterunited.json')),
    'https://www.instagram.com/chelseafc/?__a=1':
        StubMockup(text=get_file_text('data/input/instagramCrawler/chelseafc.json'))
}


class TestInstagramCrawler(TestCase):

    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._posts = []
        self._author = None
        self.instagram_crawler = InstagramCrawler(self._db)
        self.instagram_crawler.insta_crawler.get_json = types.MethodType(f, self.instagram_crawler.insta_crawler)

    def tearDown(self):
        self._db.session.close()
        pass

    def testCrawlUsers(self):
        self.instagram_crawler.crawl_users(['manchesterunited'])
        self.assertEqual(len(self._db.get_posts()), 100)
        self.assertEqual(len(self._db.get_all_authors()), 1)


if __name__ == "__main__":
    # code, test_s, test_c, test_a = Submission_Creator('https://redd.it/8a5pf1')
    # print(code)
    unittest.main()
