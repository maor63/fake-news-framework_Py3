import unittest
from configuration.config_class import getConfig
from commons.commons import *
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api
from preprocessing_tools.create_authors_table import CreateAuthorTables
from preprocessing_tools.xml_importer import XMLImporter
from commons.consts import Author_Type, DB_Insertion_Type, Author_Connection_Type
from bad_actors_collector.bad_actors_collector import BadActorsCollector
from Twitter_API.twitter_api_requester import TwitterApiRequester
from DB.schema_definition import *

class TestBadActorCollector(unittest.TestCase):
    def setUp(self):
        self.config = getConfig()
        self.db = DB()
        self.db.setUp()
        self.social_network_crawler = Twitter_Rest_Api(self.db)
        self.xml_importer=XMLImporter(self.db)
        self.create_author_table = CreateAuthorTables(self.db)
        self._targeted_twitter_author_ids = self.config.eval('BadActorsCollector', "targeted_twitter_author_ids")
        self._domain = 'Microblog'

        self._targeted_twitter_post_ids = self.config.eval('BadActorsCollector', "targeted_twitter_post_ids")
        self._bad_actor_collector = BadActorsCollector(self.db)
        self._bad_actor_collector._domain = self._domain
        #The Author and Post for test_mark_missing_bad_retweeters_retrieved_from_vico
        self._author_guid1 = '05cd2e04ffaf3c5dabd03d13b63afab6'
        author = Author()
        author.name = 'TechmarketNG'
        author.domain = self._domain
        author.protected = 0
        author.author_guid = self._author_guid1
        author.author_screen_name = 'TechmarketNG'
        author.author_full_name = 'Techmarket'
        author.statuses_count = 10
        author.author_osn_id = 149159975
        author.followers_count = 12
        author.created_at = datetime.datetime.strptime('2016-04-02 00:00:00', '%Y-%m-%d %H:%M:%S')
        author.missing_data_complementor_insertion_date = datetime.datetime.now()
        author.xml_importer_insertion_date = datetime.datetime.now()
        self.db.add_author(author)


        post = Post()
        post.post_id = 'TestPost'
        post.author = 'TechmarketNG'
        post.guid = 'TestPost'
        post.url = 'TestPost'
        tempDate = '2016-05-05 00:00:00'
        day = datetime.timedelta(1)
        post.date = datetime.datetime.strptime(tempDate, '%Y-%m-%d %H:%M:%S') + day * 1
        post.domain = self._domain
        post.author_guid = self._author_guid1
        post.content = "InternetTV love it RT @benny_metanya #wow"
        post.xml_importer_insertion_date = datetime.datetime.now()
        self.db.addPost(post)

        self.db.commit()

    def test_bad_actor_collector_not_overwriting_XML_importer(self):
        self.xml_importer.setUp()
        self.xml_importer.execute(getConfig().eval("DEFAULT","start_date"))
        self.create_author_table.setUp()
        self.create_author_table.execute(getConfig().eval("DEFAULT","start_date"))
        self._bad_actor_collector.execute()
        res = self.db.get_author_by_author_guid(compute_author_guid_by_author_name('adoula01'))
        author = res[0]
        self.assertTrue(author.xml_importer_insertion_date != None and author.bad_actors_collector_insertion_date != None)
        self.db.session.close()

    def test_BadActor_collector_crawl_bad_actors_followers(self):
        self._bad_actor_collector.crawl_bad_actors_followers()
        followers = [connection for connection in self.db.get_author_connections_by_connection_type("follower")]
        cursor = self.db.get_temp_author_connections()
        temp_author_connection_tuples = [temp_connection for temp_connection in self.db.result_iter(cursor)]
        count_total_followers = len(self.social_network_crawler._twitter_api_requester.get_follower_ids_by_user_id(714689542092169216))
        self.assertEqual(len(followers)+len(temp_author_connection_tuples),count_total_followers)
        self.db.session.close()

    def test__mark_missing_bad_retweeters_retrieved_from_vico(self):
        self._bad_actor_collector.mark_missing_bad_retweeters_retrieved_from_vico()
        test = False
        author = self.db.get_author_by_author_guid(self._author_guid1)
        if(author.mark_missing_bad_actor_retweeters_insertion_date != None and author.author_type == Author_Type.BAD_ACTOR):
            test = True
        self.assertTrue(test)
        self.db.session.close()

    def test_crawl_bad_actors_retweeters(self):
        self._bad_actor_collector.crawl_bad_actors_retweeters()
        retweets = self.db.get_post_retweeter_connections_by_post_id(self._targeted_twitter_post_ids[0])
        self.assertGreater(len(retweets), 0)
        self.db.session.close()


    def testIncorectParameters(self):
        self.db.setUp()
        test = True
        temp = self._bad_actor_collector._actions
        try:
            self._bad_actor_collector._actions = ['ggg', 'crawl_bad_actors_followers']
            self._bad_actor_collector.execute()
            author_guid = compute_author_guid_by_author_name('AnikaAhammed1')
            author = self.db.get_author_by_author_guid(author_guid)
            if (author.bad_actors_collector_insertion_date == None or author.description == None):
                test = False
        except:
            test = False
        self.assertTrue(test)
        self._bad_actor_collector._actions = temp
        self.db.session.close()


    def tearDown(self):
        self.db.session.close()
        self.db.deleteDB()

if __name__ == '__main__':
    unittest.main()
