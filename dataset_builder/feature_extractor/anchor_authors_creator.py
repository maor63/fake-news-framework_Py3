# Created by aviade at 10/04/2017


import random
from DB.schema_definition import AnchorAuthor
import copy
import sys
import logging

from preprocessing_tools.abstract_controller import AbstractController


class AnchorAuthorsCreator(AbstractController):
    def __init__(self, db, custom_targeted_class_num_of_authors_dict=None):
        AbstractController.__init__(self, db)
        if custom_targeted_class_num_of_authors_dict is None:
            self._targeted_class_num_of_authors_dict = self._config_parser.eval(self.__class__.__name__, "targeted_class_num_of_authors_dict")
        else:
            self._targeted_class_num_of_authors_dict = custom_targeted_class_num_of_authors_dict


        self._targeted_class_field_name = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_name")

        self._anchor_authors = []
        self._anchor_authors_dict = {}

    def get_targeted_class_author_guids_dict(self):
        self._db.delete_anchor_authors()
        targeted_class_author_guids_dict = self._fill_trageted_class_author_guid_dictionary()
        return targeted_class_author_guids_dict

    def create_anchor_author_dictionary(self):
        try:
            self._db.delete_anchor_authors()
            targeted_class_author_guids_dict = self._fill_trageted_class_author_guid_dictionary()

            self._randomize_anchor_authors_and_save(targeted_class_author_guids_dict)

            anchor_authors_dict = self._remove_not_anchor_authors(targeted_class_author_guids_dict)

            return anchor_authors_dict, self._anchor_authors_dict
        except Exception as e:
            print('Error: {0}'.format(e.message), file=sys.stderr)
            logging.error('Error: {0}'.format(e.message))

    def _fill_trageted_class_author_guid_dictionary(self):
        targeted_class_author_guids_dict = {}

        targeted_classes = list(self._targeted_class_num_of_authors_dict.keys())

        for targeted_class in targeted_classes:
            optional_class = str(targeted_class)
            targeted_class_author_guids_dict[optional_class] = {}

        for optional_class in targeted_classes:
            try:
                cursor = self._db.get_author_guid_by_targeted_field_name_and_targeted_class(str(self._targeted_class_field_name),
                                                                                        str(optional_class))
            except:
                print("\nError: targeted class: {0} or optional class: {1} are incorrect\n".format(self._targeted_class_field_name,optional_class), file=sys.stderr)
                return {}
            targeted_field_name_and_targeted_class_tuples = self._db.result_iter(cursor)
            for tuple in targeted_field_name_and_targeted_class_tuples:
                author_guid = tuple[0]
                targeted_class_author_guids_dict[optional_class][author_guid] = author_guid
        return targeted_class_author_guids_dict


    def _randomize_anchor_authors_and_save(self, targeted_class_author_guids_dict):
        for targeted_class, num_of_authors in self._targeted_class_num_of_authors_dict.items():
            total_targeted_class_author_guids = list(targeted_class_author_guids_dict[targeted_class].values())
            if(total_targeted_class_author_guids == []):
                raise Exception("There are no authors classified as '{0}' at DB".format(targeted_class))
            anchor_author_guids = list(random.sample(total_targeted_class_author_guids, num_of_authors))

            self._fill_anchor_authors(anchor_author_guids, targeted_class)
            self._db.add_authors(self._anchor_authors)

    def _fill_anchor_authors(self, random_targeted_class_actors, targeted_class):
        for author_guid in random_targeted_class_actors:
            anchor_author = self._create_anchor_author(author_guid, targeted_class)
            self._anchor_authors.append(anchor_author)

            self._anchor_authors_dict[author_guid] = author_guid

    def _create_anchor_author(self, author_guid, targeted_class):
        anchor_author = AnchorAuthor(author_guid, str(targeted_class))
        return anchor_author

    def _remove_not_anchor_authors(self, targeted_class_author_guids_dict):
        anchor_authors_dict = copy.deepcopy(targeted_class_author_guids_dict)
        for targeted_class, author_guids_dict in targeted_class_author_guids_dict.items():
            for labeled_author_osn_id,labeled_author_guid in author_guids_dict.items():
                if labeled_author_guid not in self._anchor_authors_dict:
                    del anchor_authors_dict[targeted_class][labeled_author_osn_id]
        return anchor_authors_dict



