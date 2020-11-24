from unittest import TestCase

from DB.schema_definition import *
from dataset_builder.lda_topic_model import LDATopicModel
from preprocessing_tools.tsv_importer import TsvImporter


class TestLDATopicModel(TestCase):
    def setUp(self):
        self._configInst = getConfig()

        self._db = DB()
        self._db.setUp()
        self._importer = TsvImporter(self._db)
        self._importer.setUp()
        self._importer.execute()  # does nothing!
        self._db.insert_or_update_authors_from_posts("Microblog", {}, {})
        self._lda = LDATopicModel(self._db)
        self._lda.setUp()

    def tearDown(self):
        self._lda.tearDown()
        self._importer.tearDown()
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()

    def testDoubleExecute(self):
        self._lda.execute(date('2015-04-27 00:00:00'))
        tcount1 = self._db.get_number_of_topics()
        self.assertEqual(tcount1, 10)

        self._lda.execute(date('2015-04-27 00:00:00'))
        tcount2 = self._db.get_number_of_topics()
        self.assertEqual(tcount1, tcount2)
        self._db.session.close()

    def testWindows(self):
        self._lda.execute(date('2015-04-26 00:00:00'))
        tcount = self._db.get_number_of_topics()
        self.assertEqual(tcount, 10)
        self._db.session.close()

    def testCleanUp(self):
        self._lda.execute(date('2015-04-26 00:00:00'))
        tcount = self._db.get_number_of_topics()
        self.assertEqual(tcount, 10)

        self._lda.cleanUp(date('2015-05-28 00:00:00'))
        tcount = self._db.get_number_of_topics()
        self.assertEqual(tcount, 0)
        self._db.session.close()

    def test_topic_size_is_correct(self):
        self._lda.execute(date('2015-04-26 00:00:00'))
        topics = self._db.get_topics()
        topic1 = [x for x in topics if x[0] == 1]
        self.assertEqual(12, len(topic1))
