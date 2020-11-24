
from preprocessing_tools.csv_importer import CsvImporter
from commons.commons import *
import csv

from preprocessing_tools.kaggle_importers.kaggle_importer_parent import Kaggle_Importer_Parent
import pandas as pd

class Kaggle_Fake_News_Importer(Kaggle_Importer_Parent):
    def __init__(self, db):
        CsvImporter.__init__(self, db)
        self._post_is_author = self._config_parser.eval(self.__class__.__name__, "post_is_author")
        self._author_type = str(self._config_parser.get(self.__class__.__name__, "author_type"))


    def parse_csv(self, csv_file, f):
        # df = pd.read_csv(self._data_folder + csv_file)
        # df = df.fillna(-1)
        # df.to_csv(self._data_folder + 'fake_new.csv')

        reader = csv.DictReader(f, delimiter=',')
        if self._post_is_author:
            self._parse_posts_as_authors(reader, csv_file)
        else:
            self._parse_posts_and_authors(reader, csv_file)

    def _parse_posts_as_authors(self, reader, csv_file):
        for row in reader:
            try:
                post_dict = {}
                text = row["text"]
                text = str(text, errors='replace')
                # content = unicode(row["text"].decode('utf-8'))
                content = text
                post_dict["content"] = content

                title = str(row["title"].decode('utf-8'))
                post_dict["title"] = title

                raw_pulished_date = str(row["published"])
                published_date = str((raw_pulished_date)[0:19].replace("/", "-").replace("T", " "))
                post_dict["date"] = published_date

                uuid = str(row["uuid"])
                post_dict["post_id"] = uuid
                post_dict["guid"] = uuid
                author_name = str(row["author"].decode('utf-8'))
                site_url = str(row["site_url"])

                if self._post_is_author:
                    post_dict["author"] = author_name + "/" + uuid
                else:
                    post_dict["author"] = site_url

                post_dict["author_guid"] = uuid

                # author_guid = unicode(compute_author_guid_by_author_name(post_dict["author"]).replace('-', ''))

                post_dict["references"] = ""
                post_dict["domain"] = self._domain
                post_dict["url"] = str("{0}/{1}".format(site_url, uuid))
                post_sub_type = str(row["type"])
                post_dict["post_type"] = post_sub_type + "/" + self._author_type

                author_name = post_dict["author"]

                author_guid = post_dict["author_guid"]

                language = str(row["language"])
                self._add_property_to_author_prop_dict(author_guid, 'language', language)

                country = str(row["country"])
                self._add_property_to_author_prop_dict(author_guid, 'location', country)
                self._add_property_to_author_prop_dict(author_guid, 'url', site_url)
                self._add_property_to_author_prop_dict(author_guid, 'created_at', raw_pulished_date)
                self._add_property_to_author_prop_dict(author_guid, 'author_osn_id', uuid)

                self._author_classify_dict[author_name] = self._author_type
                self._listdic.append(post_dict.copy())

            except:
                e = sys.exc_info()[0]
                print("Unexpected error:", e.message)
                self.logger.error("Cant encode the post:{0}".format(csv_file))
                pass

    def _add_property_to_author_prop_dict(self, author_guid, key, value):
        if value is not None:
            if author_guid not in self._author_prop_dict:
                self._author_prop_dict[author_guid] = {}
            self._author_prop_dict[author_guid][key] = value

    def _parse_posts_and_authors(self, reader, csv_file):
        author_name_author_guid_dict = {}
        for row in reader:
            try:
                post_dict = {}
                text = row["text"]
                text = str(text, errors='replace')
                # content = unicode(row["text"].decode('utf-8'))
                content = text
                post_dict["content"] = content

                title = str(row["title"].decode('utf-8'))
                post_dict["title"] = title

                raw_pulished_date = str(row["published"])
                published_date = str((raw_pulished_date)[0:19].replace("/", "-").replace("T", " "))
                post_dict["date"] = published_date

                uuid = str(row["uuid"])
                post_dict["post_id"] = uuid
                post_dict["guid"] = uuid
                target_author_name = str(row["author"].decode('utf-8'))
                site_url = str(row["site_url"])

                main_img_url = str(row["main_img_url"])
                post_dict["source_url"] = main_img_url

                if len(target_author_name) == 0:
                    target_author_name = uuid

                elif target_author_name is 'Anonymous':
                    target_author_name = target_author_name + "/" + uuid

                author_guid = compute_author_guid_by_author_name(target_author_name)

                post_dict["author"] = target_author_name


                post_dict["author_guid"] = author_guid

                # author_guid = unicode(compute_author_guid_by_author_name(post_dict["author"]).replace('-', ''))

                post_dict["references"] = ""
                post_dict["domain"] = self._domain
                post_dict["url"] = str(site_url)
                post_sub_type = str(row["type"])
                post_dict["post_type"] = post_sub_type + "/" + self._author_type

                language = str(row["language"])
                self._add_property_to_author_prop_dict(author_guid, 'language', language)

                country = str(row["country"])
                self._add_property_to_author_prop_dict(author_guid, 'location', country)
                self._add_property_to_author_prop_dict(author_guid, 'url', site_url)
                self._add_property_to_author_prop_dict(author_guid, 'created_at', raw_pulished_date)
                self._add_property_to_author_prop_dict(author_guid, 'author_osn_id', uuid)

                self._author_classify_dict[target_author_name] = self._author_type
                self._listdic.append(post_dict.copy())

            except:
                e = sys.exc_info()[0]
                print("Unexpected error:", e.message)
                self.logger.error("Cant encode the post:{0}".format(csv_file))
                pass
