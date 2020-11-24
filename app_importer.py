# encoding: utf-8
#need to be added to the system

import codecs
from os import listdir
from preprocessing_tools.post_importer import PostImporter
from configuration.config_class import getConfig
from DB.schema_definition import *
from commons.commons import compute_author_guid_by_author_name

class AppImporter(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        config_parser = getConfig()
        self.start_date = config_parser.eval("DEFAULT","start_date")
        self.end_date = config_parser.eval("DEFAULT", "end_date")
        self._data_folder = self._config_parser.eval(self.__class__.__name__, "data_folder")
        self._bad_actor_threshold = self._config_parser.eval(self.__class__.__name__, "bad_actor_threshold")

    def readFromFolders(self):
        all_apps = listdir(self._data_folder)
        i = 1
        author_classify_dict = {}
        for app in all_apps:
            if(i % 100 == 0):
                print(("Import app [{0}/{1}]".format(i,len(all_apps))))
            i+=1
            f = codecs.open(self._data_folder+app, "r",encoding='CP1252')
            rows = f.readlines()
            if(int(app[0]) > self._bad_actor_threshold):
                author_classify_dict[app[2:len(app) - 4]] = "bad_actor"
            else:
                author_classify_dict[app[2:len(app) - 4]] = "good_actor"
            for row in rows:
                try:
                    guid = generate_random_guid()
                    post_dict = {}
                    post_dict["author_guid"] = str(app[2:len(app) - 4])
                    post_dict["content"] = str(cleaner(row[:-1]))
                    post_dict["date"] = str(self.start_date)
                    post_dict["guid"] = guid
                    post_dict["author"] = str(app[2:len(app) - 4])
                    post_dict["references"] = ""
                    post_dict["domain"] = "app"
                    post_dict["url"] = "apps.com/"+ guid
                    post_dict["author_osn_id"] = str(app[2:len(app) - 4])
                    self._listdic.append(post_dict.copy())
                except:
                    self.logger.error("Cant encode the post:{0}".format(app))
                    pass
            f.close()
        print ("Insert posts to DB")
        self.insertPostsIntoDB()
        print("Insert Authors")
        self._db.insert_or_update_authors_from_posts("app", author_classify_dict)




db = DB()
db.setUp()
a= AppImporter(db)
a.readFromFolders()
