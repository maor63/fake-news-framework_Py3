# Created by aviade      
# Time: 02/05/2016 15:00


import logging
from configuration.config_class import getConfig
from DB.schema_definition import *
from os import listdir
import urllib.request, urllib.parse, urllib.error
from xml.dom.minidom import parse
from xml.parsers.expat import ExpatError
from commons.commons import *
import os
import timeit

from preprocessing_tools.post_importer import PostImporter


# from _mysql import NULL

class XMLImporter(PostImporter):
    def __init__(self, db):

        PostImporter.__init__(self, db)
        config_parser = getConfig()
        self.xmlPath = config_parser.get(self.__class__.__name__, "xml_path")

        # self.xmlPath = configInst.get(self.__class__.__name__,"XMDL_source_path")
        self.fileName = None
        self.CurrFolderPath = None

    def setUp(self):
        logging.info("setUp")
        logging.info("readFromFolders")
        self.readFromFolders()

    def execute(self, window_start=None):
        logging.info("execute")
        logging.info("PostImporter.execute(self, window_start)")
        PostImporter.execute(self, window_start)
        # self.readFromFolders()

    def canProceedNext(self, window_start):
        return True

    def cleanUp(self, window_start):
        pass

    def readFromFolders(self):

        startreadfromfolder = timeit.default_timer()

        allSubFolders = listdir(self.xmlPath)

        for currfolder in allSubFolders:

            if os.path.isfile(self.xmlPath + currfolder):
                self.CurrFolderPath = self.xmlPath

                self._listdic = self.parseXMLsToListdict(self.CurrFolderPath)
                self.insertPostsIntoDB()
                break

            if os.path.isdir(self.xmlPath + currfolder):
                self.CurrFolderPath = self.xmlPath + currfolder + '/'

                self._listdic = self.parseXMLsToListdict(self.CurrFolderPath)
                self.insertPostsIntoDB()
        stopreadfromfolder = timeit.default_timer()
        print("parse and read ", stopreadfromfolder - startreadfromfolder)

    def parseXMLsToListdict(self, xmlPath):
        print("parseXMLsToListdict")
        startparsetolistdict = timeit.default_timer()

        self.xmlPath = xmlPath
        listdic = []
        urls = []
        logging.info("total XML: " + str(len(os.listdir(self.xmlPath))))
        i = 1
        for xmlFile in os.listdir(self.xmlPath):
            msg = "\r Parse XML: [{}".format(i) + "/" + str(len(os.listdir(self.xmlPath))) + ']'
            print(msg, end="")
            # logging.info("Parse: "+str(i)+"/"+str(len(os.listdir(self.xmlPath))))
            self._listdic = self.parseXMLToListdict(xmlFile, listdic, urls)
            i += 1

        stopparsetolistdict = timeit.default_timer()

        print("\nsingle file parser ", stopparsetolistdict - startparsetolistdict)
        return self._listdic

    def parseXMLToListdict(self, xmlFile, listdic, urls):
        file_path = str(self.xmlPath + xmlFile)
        f = open(file_path)
        try:
            dom = parse(f)
            for doc in dom.getElementsByTagName('ns2:doc'):
                dictionary = {}
                doc_children = doc._get_childNodes()
                linklist = []
                haslinkes = "false"
                for doc_child in doc_children:
                    author = False
                    if doc_child.tagName == 'links':
                        allSplitedLinks = (doc_child.firstChild.data).split(' ')
                        for link in allSplitedLinks:
                            normalized_link = normalize_url(cleaner(link))
                            linklist.append(normalized_link)
                        haslinkes = "true"
                    try:
                        if doc_child.tagName == 'article':
                            article_children = doc_child._get_childNodes()
                            for art_child in article_children:
                                # TODO: replace try except with appropriate conditions.

                                try:
                                    if art_child.tagName == 'document_guid':
                                        dictionary.update({'guid': cleaner(art_child.firstChild.data)})
                                except:
                                    self.logger.error("missing document_guid in file: {0}".format(file_path))
                                try:
                                    if art_child.tagName == 'headline':
                                        dictionary.update({'title': cleaner(art_child.firstChild.data)})
                                except:
                                    self.logger.error("missing headline in file: {0}".format(file_path))
                                try:
                                    if art_child.tagName == 'fulltext':
                                        content = art_child.getElementsByTagName('paragraph')[0]
                                        dictionary.update({'content': cleaner(content.firstChild.data)})
                                except:
                                    self.logger.error("missing fulltext in file: {0}".format(file_path))
                                try:
                                    if art_child.tagName == 'author':
                                        dictionary.update(
                                            {'author': createunicodedata(cleanForAuthor(art_child.firstChild.data))})
                                        author = True
                                except:
                                    # the author will be NULL
                                    author = False
                                    self.logger.error("missing author in file: {0}".format(file_path))

                                try:
                                    if art_child.tagName == 'authorGuid':
                                        dictionary.update({'author_guid': art_child.firstChild.data})
                                        if (author == False):
                                            dictionary.update({'author': cleanForAuthor(art_child.firstChild.data)})
                                except:
                                    self.logger.error("missing author GUID in file: {0}".format(file_path))

                                try:
                                    if art_child.tagName == 'url':
                                        url = art_child.firstChild.data
                                        normalized_url = normalize_url(url)
                                        dictionary.update({'url': normalized_url})
                                except:
                                    self.logger.error("missing url in file: {0}".format(file_path))

                                try:
                                    if art_child.tagName == 'published_datetime':
                                        dictionary.update({'date': cleaner(art_child.firstChild.data)})
                                except:
                                    self.logger.error("missing published_datetime in file: {0}".format(file_path))

                    except:
                        self.logger.error("missing article file name in file: {0}".format(file_path))

                    try:
                        if doc_child.tagName == 'source':
                            source_children = doc_child._get_childNodes()
                            for source_child in source_children:
                                if source_child.tagName == 'subcategory':
                                    dictionary.update(
                                        {'domain': createunicodedata(cleaner(source_child.firstChild.data))})
                    except:
                        self.logger.error("no source in file: {0}".format(file_path))

                if (haslinkes == "true"):

                    dictionary.update({'references': '|'.join(linklist)})

                else:
                    dictionary.update({'references': ''})

                isThisPostURLUnique = self.filterUniqueURL(urls, dictionary)
                if isThisPostURLUnique:
                    listdic.append(dictionary.copy())

                dictionary.clear()
        except ExpatError:
            self.logger.error("bad XML")

        return listdic

    def filterUniqueURL(self, urls, dict):
        url = dict.get('url')
        if url not in urls:
            urls.append(url)
            return True
        return False


if __name__ == "__main__":
    pass

########################################################################
import unittest


class test_xml_importer(unittest.TestCase):
    def setUp(self):
        db = DB()
        self.xml_importer = XMLImporter(db)

    def test_umlaut_chars_in_authors_names(self):
        '''
        Tests that authors' names that contain umlaut are encoded properly, i.e. o with umlaut => o, u wit umlaut => u etc.
        '''
        config_parser = getConfig().get_config_parser()
        xmlPath = config_parser.get("XMLImporter", "xml_source_path")
        listdic = self.xml_importer.parseXMLsToListdict(xmlPath)
        author_name = listdic[0]['author']
        self.assertFalse('\xf6' in author_name, "Author name contain umlaut - unlaut wasn't encoded properly")
