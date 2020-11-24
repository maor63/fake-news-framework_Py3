# Created by Jorge Bendahan (jorgeaug@post.bgu.ac.il) at 26/06/2016
# Ben Gurion University of the Neguev - Department of Information Systems Engineering

import logging
from commons.data_frame_creator import DataFrameCreator
# encoding=utf8
import sys
import imp
# imp.reload(sys)
# sys.setdefaultencoding('utf8')

'''
This class is responsible reading the features from author_features table, transforming the data
into the following structure: author_guid, feature_1, ..., feature_n
and finally writing this data into a ARFF file
'''


class ArffWriter():
    def __init__(self, db, author_type_classes, author_sub_type_classes, target_type_attr_name):
        self._db = db
        self._author_type_classes = author_type_classes
        self._author_subtype_classes = author_sub_type_classes
        self._target_type_attr_name = target_type_attr_name

    def setUp(self):
        pass

    def write_author_features_to_arff(self, output_filename):
        data_frame_creator = DataFrameCreator(self._db)

        data_frame_creator.create_author_features_data_frame()
        author_features_data_frame = data_frame_creator.get_author_features_data_frame()

        author_features_data_frame = data_frame_creator.fill_empty_fields_for_dataframe(author_features_data_frame)

        data_file = open(output_filename, 'w')

        logging.info("Start writing ARFF file")

        header = '@RELATION bad_actors \n '
        header += '@ATTRIBUTE author_guid string \n '
        for col in author_features_data_frame.columns:
            if col == "author_screen_name" or col == "AccountPropertiesFeatureGenerator_author_screen_name":
                header += '@ATTRIBUTE ' + col + ' string \n '
            elif col == "author_type" or col == "AccountPropertiesFeatureGenerator_author_type":
                author_type_str = ' {'
                for author_type_class in self._author_type_classes:
                    author_type_str = author_type_str + author_type_class + ", "
                author_type_str = author_type_str[:-2]
                author_type_str = author_type_str + "} \n"
                header += '@ATTRIBUTE '+col+ author_type_str
                #header += '@ATTRIBUTE '+col+' {bad_actor, good_actor} \n'

            elif col == "author_sub_type":
                if len(self._author_subtype_classes) > 0:
                    author_type_str = ' {'
                    for author_type_class in self._author_subtype_classes:
                        author_type_str = author_type_str + author_type_class + ", "
                    author_type_str = author_type_str[:-2]
                    author_type_str = author_type_str + "} \n"
                    header += '@ATTRIBUTE ' + col + author_type_str
                else:
                    header += '@ATTRIBUTE ' + col + ' string \n '
            elif col == self._target_type_attr_name:
                header += '@ATTRIBUTE ' + col + ' {' + ','.join(map(str, self._author_type_classes)) + '} \n'
            else:
                header += '@ATTRIBUTE '+col+' numeric \n '

        header += ' @DATA \n '
        data_file.write(header)

        records = author_features_data_frame.to_records()
        total_authors = len(records)
        curr_author = 0
        for record in records:
            curr_author += 1
            #print(" " + record[0])
            print ('\r writing author '+str(curr_author)+' of '+str(total_authors), end="")
            str_record = ''.join([str(record[i]).encode('utf-8') + ', ' for i in range(0, len(record)-1)])
            str_record += str(record[len(record) - 1]).encode('utf-8') + ' \n'
            data_file.write(str_record)
        data_file.close()

        logging.info("Finished writing ARFF file")