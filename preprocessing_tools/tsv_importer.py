__author__ = 'users'


import logging
from configuration.config_class import getConfig
from commons import *
from DB.schema_definition import *
from os import listdir
import urllib.request, urllib.parse, urllib.error
import csv
from preprocessing_tools.post_importer import PostImporter
import datetime
from datetime import datetime
from commons.commons import normalize_url

"""
@description: read data from .csv file or files tree
and insert it to .DB file
"""
class TsvImporter(PostImporter):
    
    def __init__(self,db):
        
        PostImporter.__init__(self,db)
        
        configInst = getConfig()

        self.source_path = configInst.get(self.__class__.__name__,"FDL_source_path")
        self.file_date_format = configInst.get(self.__class__.__name__,"date_format")
        self.fileName = None #@review: not a field. make it local variable. 
        self.URLforUnittest = None #@review: no code should be written especially for unittests (except the tests)
        self.CurrFolderPath = None #@review: coding convention: small letter field names and _ for private members 
        


    def setUp(self):
        """
        @description: if FDL_read_from_tree=true the setUp will
         read documents from all files in folder tree
        """
        #self.CheckFilesTypeValidity()
        allSubFolders = listdir(self.source_path)
        for currfolder in allSubFolders:
            
            if os.path.isfile(self.source_path + currfolder):
                self.CurrFolderPath= self.source_path #@review: CurrFolderPath should be a local variable
                sourceFilesList = []
                sourceFilesList.append(currfolder)
            if os.path.isdir(self.source_path + currfolder):
                
                self.CurrFolderPath = self.source_path + currfolder + '/'
                sourceFilesList = listdir(self.source_path + currfolder)
                
            if len(sourceFilesList)==0:
                self.logger.error("there isn't any file in source folder {0}".format(self.source_path))
            else:
                for file in sourceFilesList:
                    if file.endswith('.csv'):
                        self.fileName = file

            
                    listdic = []
                    self.parseTsvToListDict(listdic)
                    self.insertPostsIntoDB()




    def execute(self,window_start=None):
        """
        precondition: None
        """
        pass

    def canProceedNext(self,window_start):
        return True


    def parseTsvToListDict(self,listdic):
        
        
        f = open(self.CurrFolderPath+"\\"+self.fileName)
        logging.info("IMPORT CSV %s"%f)
        try:
            urls = []
            reader = csv.DictReader(f,delimiter='\t')
            for row in reader:
#                 cells = row.split('\t') #@use: DictReader to read tsv files (https://docs.python.org/2/library/csv.html)  
#                 #need to think about general method for this
#                 Guid,Title,URL,Date,Content,Source_Type,Language,Community,Author,Thread,Sentiment,Research_Object,Snippet, \
#                 Country,Score,Reach,Entries,Relevance,Alexa_Rank,Backlinks,Likes,Shares,Views,Comments,isComment,isShare,Links = cells
                Guid = row['Guid']
                Title = row['Title']
                URL = row['URL']
                Date = row['Date']
                Content = row['Content']
                Source_Type = row['Source Type']
                Author = row['Author']
                Author_GUID = row['Author GUID']
                Links = row['Links']
                dictionary = {}
                linklist = []
                haslinkes = "false"
                if Links!='\n':
                    
                    allSplitedLinks = (Links).split(' | ')
                    for link in allSplitedLinks:
                        
                        linklist.append(normalize_url(str(cleaner(link),'utf-8',errors='ignore')))
                        
                    haslinkes = "true"

                dictionary.update({'guid':str(cleaner(Guid),'utf-8',errors='ignore')})
                dictionary.update({'title':str(cleaner(Title),'utf-8',errors='ignore')})
                dictionary.update({'content':str(cleaner(Content),'utf-8',errors='ignore')})
                author = str(cleanForAuthor(Author),'utf-8',errors='ignore')
                dictionary.update({'author':author})
                dictionary.update({'author_guid':str(cleaner(Author_GUID),'utf-8',errors='ignore')})
                unicodeURL = normalize_url(str(cleaner(URL),'utf-8',errors='ignore'))

                dictionary.update({'url':unicodeURL})
                    
                self.URLforUnittest = unicodeURL
                
                Date = datetime.strptime(Date, self.file_date_format).date()
                Date = date_to_str(Date) 
                dictionary.update({'date':cleaner(Date)})
                dictionary.update({'domain':str(cleaner(Source_Type),'utf-8',errors='ignore')})


                if (haslinkes=="true"):
                    allLinks = '|'.join(linklist)
                    dictionary.update({'references':allLinks})
                else:
                    dictionary.update({'references':''})

                #validate that the same url not inserted to listdic twice
                isThisPostURLUnique = self.filterUniqueURL(urls,dictionary)
                if isThisPostURLUnique:
                    listdic.append(dictionary.copy())

                dictionary.clear()

        except:
            logging.exception( "ERROR: can't read file: ")
        finally:
            f.close()
        self._listdic = listdic

    def filterUniqueURL(self,urls,dict):
        
        url = dict.get('url')
        if url not in urls:
            urls.append(url)
            return True
        return False

    def cleanUp(self,window_start):
        pass


########################################################################
import sys
import unittest

class TestCSVDataImport(unittest.TestCase):
    def setUp(self):
        import sys
        sys.argv = [sys.argv[0], 'config_test_offline.ini']

        self._db = DB()
        self._db.setUp()
        self._fdlObject = TsvImporter(self._db)
        self._fdlObject.setUp()

    def tearDown(self):
        #pass
        self._db.tearDown()

    def testAssureConfig(self):
        self.assertEqual(sys.argv[1], 'config_test_offline.ini')

    def testRowExist(self):
        self.assertTrue(self._fdlObject.canProceedNext(date('2015-04-27 00:00:00')),
                        "the post " + self._fdlObject.URLforUnittest + " is not inserted into the DB")

    def testAddedAllRecordsToDB(self):
        #works with test_offline_16
        max_date = self._db.getPostsMaxDate()
        min_date = self._db.getPostsMinDate()
        self.assertEqual(len(self._db.getPostsListWithoutEmptyRowsByDate(min_date,max_date)), 20, "not all records were inserted")

        tcount = self._db.session.execute("select count(*) from post_citations").scalar()
        self.assertEqual(tcount, 36)
        


if __name__ == "__main__":
    unittest.main()