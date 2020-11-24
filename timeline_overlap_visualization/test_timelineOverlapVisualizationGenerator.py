from unittest import TestCase
from DB.schema_definition import DB, Author, Post
from configuration.config_class import getConfig
import datetime
from timeline_overlap_visualization.timeline_overlap_visualization_generator import TimelineOverlapVisualizationGenerator


class TestTimelineOverlapVisualizationGenerator(TestCase):
    def setUp(self):
        self.config = getConfig()
        self._db = DB()
        self._db.setUp()
        self.timeline_overlap = TimelineOverlapVisualizationGenerator()

        author1 = Author()
        author1.name = 'acquired_user'
        author1.domain = 'Microblog'
        author1.author_guid = 'acquired_user'
        author1.author_screen_name = 'acquired_user'
        author1.author_full_name = 'acquired_user'
        author1.author_osn_id = 1
        author1.created_at = datetime.datetime.now()
        author1.missing_data_complementor_insertion_date = datetime.datetime.now()
        author1.xml_importer_insertion_date = datetime.datetime.now()
        author1.author_type = 'bad_actor'
        author1.author_sub_type = 'acquired'
        self._db.add_author(author1)

        for i in range(1,11):
            post1 = Post()
            post1.post_id = 'bad_post'+str(i)
            post1.author = 'acquired_user'
            post1.guid = 'bad_post'+str(i)
            post1.date = datetime.datetime.now()
            post1.domain = 'Microblog'
            post1.author_guid = 'acquired_user'
            post1.content = 'InternetTV love it'+str(i)
            post1.xml_importer_insertion_date = datetime.datetime.now()
            self._db.addPost(post1)

        author = Author()
        author.name = 'TestUser1'
        author.domain = 'Microblog'
        author.author_guid = 'TestUser1'
        author.author_screen_name = 'TestUser1'
        author.author_full_name = 'TestUser1'
        author.author_osn_id = 2
        author.created_at = datetime.datetime.now()
        author.missing_data_complementor_insertion_date = datetime.datetime.now()
        author.xml_importer_insertion_date = datetime.datetime.now()
        self._db.add_author(author)

        for i in range(1, 11):
            post = Post()
            post.post_id = 'TestPost'+str(i)
            post.author = 'TestUser1'
            post.guid = 'TestPost'+str(i)
            post.date = datetime.datetime.now()
            post.domain = 'Microblog'
            post.author_guid = 'TestUser1'
            post.content = 'InternetTV love it'+str(i)
            post.xml_importer_insertion_date = datetime.datetime.now()
            self._db.addPost(post)

        self._db.commit()

    def test_generate_timeline_overlap_csv(self):
        self.timeline_overlap.setUp()
        self.timeline_overlap.generate_timeline_overlap_csv()
        author = self._db.get_author_by_author_guid('acquired_user')
        self.assertEqual(author.author_type, 'bad_actor')
        self.assertEqual(author.author_sub_type, 'acquired')
        pass

    def tearDown(self):
        self._db.session.close_all()
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()
