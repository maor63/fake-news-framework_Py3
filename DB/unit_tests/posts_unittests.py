# Created by aviade      
# Time: 03/05/2016 16:12

from DB.schema_definition import Post, Author
from .test_base import TestBase
from commons.commons import *

class TestPost(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.post_id = str(generate_random_guid())
        self.author_guid = str(generate_random_guid())
        self.new_author_guid = str(generate_random_guid())



    def testDBSetUp(self):
        TestBase.testDBSetUp(self)
        self.assertTrue("posts" in set(self.inspector.get_table_names()))
        self.clean()


    ########################################################################
    # posts's unit tests
    ########################################################################

    def create_dummy_post(self):
        post = Post()

        post.post_id = str(self.post_id)
        post.author = "author"
        post.guid = str(generate_random_guid())
        post.title = "title"
        post.url = "http://google.com"
        post.date = str_to_date("2016-08-24 10:00:15")
        post.content = "text"
        post.is_detailed = True
        post.is_LB = False
        post.is_valid = True
        post.domain = "Google"
        post.author_guid = str(self.author_guid)
        post.post_osn_id = 123455678
        post.retweet_count = 11
        post.favorite_count = 10
        post.created_at = "2016-08-24 10:00:15"

        return post

    def create_and_insert_post(self):
        post = self.create_dummy_post()
        self.db.addPost(post)
        self.db.session.commit()
        return post

    def testInsertPost(self):
        self.setup()

        self.create_and_insert_post()

        selected_post = self.db.get_post_by_id(str(self.post_id))
        self.assertEqual(self.post_id, selected_post.post_id)
        self.assertEqual(self.author_guid, selected_post.author_guid)

        self.db.delete_post(self.post_id)
        self.clean()


    def testDeletePost(self):
        self.setup()

        self.create_and_insert_post()

        self.db.delete_post(self.post_id)

        result = self.db.get_post_by_id(str(self.post_id))
        self.assertEqual(None, result)
        self.clean()

    def testUpdatePost(self):
        self.setup()

        post = self.create_and_insert_post()

        #self.db.update_post(self.previous_text, "text", self.new_text)
        post.author_guid = str(self.new_author_guid)
        self.db.addPost(post)

        selected_post = self.db.get_post_by_id(str(self.post_id))
        self.assertEqual(self.new_author_guid, selected_post.author_guid)

        self.db.delete_post(self.post_id)
        self.clean()

