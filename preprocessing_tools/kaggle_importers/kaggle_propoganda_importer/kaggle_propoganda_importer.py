# Created by Aviad Elyashar at 23/04/2017

import csv
import traceback
import logging
from preprocessing_tools.kaggle_importers.kaggle_importer_parent import Kaggle_Importer_Parent
from commons.commons import *
from datetime import datetime

class Kaggle_Propoganda_Importer(Kaggle_Importer_Parent):
    def __init__(self, db):
        Kaggle_Importer_Parent.__init__(self, db)
        self._author_type = str(self._config_parser.get(self.__class__.__name__, "author_type"))
        self._post_url = "Kaggle_Propoganda"

    def parse_csv(self, csv_file, f):

        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            try:
                post_dict = {}

                username = str(row["username"].decode('utf-8'))
                author_guid = compute_author_guid_by_author_name(username)

                author_name = str(row["name"].decode('utf-8'))
                self._add_property_to_author_prop_dict(author_guid, "name", username)

                self._add_property_to_author_prop_dict(author_guid, "name", author_name)


                post_dict["author"] = username
                self._add_property_to_author_prop_dict(author_guid, "author_screen_name", username)
                self._add_property_to_author_prop_dict(author_guid, "author_full_name", author_name)
                self._add_property_to_author_prop_dict(author_guid, "author_osn_id", author_guid)

                post_dict["author_guid"] = author_guid

                description = str(row["description"].decode('utf-8'))
                post_dict["description"] = description

                location = str(row["location"].decode('utf-8'))
                self._add_property_to_author_prop_dict(author_guid, "location", location)

                num_of_followers = int(row["followers"])
                self._add_property_to_author_prop_dict(author_guid, "followers_count", num_of_followers)

                time = str(row["time"])

                publication_date = datetime.strptime(time, "%m/%d/%Y %H:%M")
                post_dict["date"] = publication_date

                post_dict["references"] = ""
                post_dict["domain"] = self._domain

                str_publication_date = date_to_str(publication_date)

                content = str(row["tweets"].decode('utf-8'))
                post_dict["content"] = content


                post_id = compute_post_guid(content, author_name, str_publication_date)
                post_dict["post_id"] = post_id
                post_dict["guid"] = post_id

                post_dict["url"] = str("{0}/{1}".format(self._post_url, post_id))

                num_of_statuses = int(row["numberstatuses"])
                self._add_property_to_author_prop_dict(author_guid, "statuses_count", num_of_statuses)

                post_sub_type = "ISIS_terrorist"
                post_dict["post_type"] = post_sub_type + "/" + self._author_type

                self._author_classify_dict[username] = self._author_type
                self._listdic.append(post_dict.copy())
                msg = "\r Post id: {} was added".format(post_dict["post_id"])
                print(msg, end="")

            except Exception as e:
                logging.error(traceback.format_exc())
