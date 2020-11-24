# Created by aviade      
# Time: 10/05/2016 11:56

########################################################################
import unittest
from DB.schema_definition import DB
from configuration.config_class import getConfig
import os

class TestDB(unittest.TestCase):
    def setUp(self):
        import sys
        sys.argv = [sys.argv[0], 'config_test_offline.ini']

        self._config= getConfig()

    def tearDown(self):
        pass

    def testDBExists(self):
        fileList = os.listdir(self._config.get("DB","DB_path"))
        self.assertTrue("SOMECDB_test_offline.db" in fileList)

    def testDBSetUp(self):
        from sqlalchemy.engine.reflection import Inspector
        db = DB()
        db.setUp()
        session = db.Session()
        inspector = Inspector.from_engine(db.engine)
        self.assertTrue("posts" in set(inspector.get_table_names()))

    def testLogarithm(self):
        db = DB()
        db.setUp()
        session = db.Session()
        recs = session.execute("select log(10) as q;")
        for rec in recs:
            self.assertAlmostEqual(rec[0],2.302585092994046)

    def test_query_XX(self):
        #@todo: add query tests
        pass

    def testDoubleExecute(self):
        import sys
        sys.argv = [sys.argv[0], 'config.ini']
        db = DB()
        db.setUp()
        db.execute(getConfig().get("DEFAULT","start_date"))
        getTablesQuerys=["select * from posts","select * from authors","select * from topics","select * from author_citations","select * from authors_boost_stats","select * from post_citations","select * from posts_representativeness","select * from posts_to_pointers_scores","select * from posts_to_topic","select * from visualization_windows"]
        listNumTablesRows=[]
        for tableQuery in getTablesQuerys:
            listNumTablesRows.append(db.session.execute(tableQuery).scalar())
        db.setUp()
        db.execute(getConfig().get("DEFAULT","start_date"))
        listNumTablesRows2=[]
        for tableQuery in getTablesQuerys:
            listNumTablesRows2.append(db.session.execute(tableQuery).scalar())
        self.assertListEqual(listNumTablesRows,listNumTablesRows2,"the two executions had different results")



