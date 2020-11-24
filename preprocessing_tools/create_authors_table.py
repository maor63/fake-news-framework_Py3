# Created by aviade      
# Time: 09/05/2016 16:25

'''
@author: Luiza Nacshon
'''
import logging
from preprocessing_tools.abstract_controller import AbstractController


class CreateAuthorTables(AbstractController):
    #TODO: 4: remove module.
    def __init__(self, db):

        AbstractController.__init__(self, db)

    def setUp(self):
        pass


    def execute(self,window_start):
        logging.info("execute")
        AbstractController.execute(self, window_start)

        logging.info("db.deleteAuthCit")
        self._db.deleteAuthCit(self._window_start)
        logging.info("db.insert_or_update_authors_from_xml_importer")
        #self._db.insertIntoAuthorsTable(self._window_start,self._window_end)
        self._db.insert_or_update_authors_from_xml_importer(self._window_start,self._window_end)
        self._db.insertIntoAuthorCitation(self._window_start,self._window_end)

        logging.info("finished execute!")

    def canProceedNext(self,window_start):
        return True


    def cleanUp(self,window_start):
        pass



    '''
import unittest
class TestCreateAuthorTables(unittest.TestCase):
    pass
    def setUp(self):
        import sys
        sys.argv = [sys.argv[0], 'config_test_offline.ini']
        self._configInst = Configuration.get_config_parser()

        from preprocessing_tools.tsv_importer import TsvImporter

        self._db = DB()
        self._db.setUp()
        self._importer = TsvImporter(self._db)
        self._importer.setUp()
        self._importer.execute()#does nothing!

        self._auth = CreateAuthorTables(self._db)
        self._auth.setUp()
        pass

    def testNone(self):
        pass

    def testExecute(self):
        self._auth.execute(date('2015-04-26 00:00:00'))
        tcount = self._db.session.execute("select count(*) from authors").scalar()
        self.assertEqual(tcount, 14)
        tcount = self._db.session.execute("select count(*) from author_citations").scalar()
        self.assertEqual(tcount, 17)
        pass

    def testDoubleExecute(self):
        self._auth.execute(date('2015-04-26 00:00:00'))
        tcount = self._db.session.execute("select count(*) from authors").scalar()
        self.assertEqual(tcount, 14)
        tcount = self._db.session.execute("select count(*) from author_citations").scalar()
        self.assertEqual(tcount, 17)

        self._auth.execute(date('2015-04-26 00:00:00'))
        tcount = self._db.session.execute("select count(*) from authors").scalar()
        self.assertEqual(tcount, 14)
        tcount = self._db.session.execute("select count(*) from author_citations").scalar()
        self.assertEqual(tcount, 17)
        pass

    def testExecuteLess(self):
        self._auth.execute(date('2015-04-28 00:00:00'))
        tcount = self._db.session.execute("select count(*) from authors").scalar()
        self.assertEqual(tcount, 14)
        tcount = self._db.session.execute("select count(*) from author_citations").scalar()

        self.assertEqual(tcount, 14) #@todo: manually validate the test
        pass

    '''