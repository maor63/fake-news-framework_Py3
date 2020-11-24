# Created by aviade      
# Time: 28/04/2016 09:30
import unittest
from DB.schema_definition import DB
#from configuration.configuration import Configuration
from configuration.config_class import getConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class TestBase(unittest.TestCase):
    '''
    def setUp(self):
        configInst = getConfig()
        self._path_to_engine = configInst.get("DB", "DB_path") + configInst.get("DB", "DB_name")

        self.engine = create_engine("sqlite:///" + self._path_to_engine, echo=False)
        session = sessionmaker()
        session.configure(bind=self.engine)
        self.session = session()
    '''

    def tearDown(self):
        self.db.session.close()

    def testDBSetUp(self):
        from sqlalchemy.engine.reflection import Inspector
        self.db = DB()
        self.db.setUp()
        self.inspector = Inspector.from_engine(self.db.engine)
        #self.assertTrue("posts" in set(self.inspector.get_table_names()))
        self.db.session.close()

    #This method initialize a DB for tests
    def setup(self):
        import sys
        sys.argv = [sys.argv[0], 'configuration/config_test.ini']
        self.db = DB()
        self.db.setUp()

    #This method clean the DB created in setup
    #Must come at the end of a test if setup has been used
    def clean(self):
        self.db.session.close()
        self.db.deleteDB()