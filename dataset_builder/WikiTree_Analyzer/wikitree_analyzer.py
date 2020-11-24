
from commons.method_executor import Method_Executor
from sframe import SFrame
import re
import pandas as pd
import editdistance
import time
import datetime
import jellyfish
import phonetics
import sframe.aggregate as agg
import urllib.request, urllib.error, urllib.parse
import json
from geopy.geocoders import Bing
from geopy.exc import GeocoderTimedOut, GeocoderQueryError, GeocoderQuotaExceeded
import pickle
import networkx as nx
import itertools
import py_common_subseq


__author__ = "Aviad Elyashar"


class WikiTreeAnalyzer(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_directory_path = self._config_parser.eval(self.__class__.__name__, "input_directory_path")
        self._target_file_name = self._config_parser.eval(self.__class__.__name__, "target_file_name")
        self._output_directory_path = self._config_parser.eval(self.__class__.__name__, "output_directory_path")
        self._target_field_name = self._config_parser.eval(self.__class__.__name__, "target_field_name")
        self._ancestor_field_name = self._config_parser.eval(self.__class__.__name__, "ancestor_field_name")
        self._results_file_name = self._config_parser.eval(self.__class__.__name__, "results_file_name")
        self._node_features = self._config_parser.eval(self.__class__.__name__, "node_features")


    def create_son_father_mother_file_by_field_name(self):
        # target fle should be dump_people_user_full.csv
        wikitree_sf = self._read_csv_file("\t")

        wikitree_sf = wikitree_sf[(wikitree_sf[self._target_field_name] != None) &
                                  (wikitree_sf[self._target_field_name] != '') &
                                  (wikitree_sf[self._target_field_name] != 'Unknown') &
                                  (wikitree_sf[self._target_field_name] != 'Anonymous')]

        print("Father's calculations are starting!")

        son_father_sf = wikitree_sf.join(wikitree_sf, on={'Father': 'User ID'}, how='inner')

        son_father_sf = son_father_sf.select_columns(
            ['User ID', 'WikiTree ID', self._target_field_name, 'Gender', 'Birth Date', 'Birth Location',
             'Death Date', 'Death Location', 'Father', 'WikiTree ID.1', self._target_field_name + ".1", 'Gender.1',
             'Birth Date.1', 'Birth Location.1', 'Death Date.1', 'Death Location.1', 'Mother'])

        son_father_sf = son_father_sf.rename(
            {'WikiTree ID.1': 'Father WikiTree ID', self._target_field_name +".1": 'Father ' + self._target_field_name,
             'Gender.1': 'Father Gender',
             'Birth Date.1': 'Father Birth Date', 'Birth Location.1': 'Father Birth Location',
             'Death Date.1': 'Father Death Date', 'Death Location.1': 'Father Death Location'})

        print("Father's calculations were finished!")

        son_father_mother_sf = son_father_sf.join(wikitree_sf, on={'Mother': 'User ID'}, how='inner')

        son_father_mother_sf = son_father_mother_sf.select_columns(
            ['User ID', 'WikiTree ID', self._target_field_name, 'Gender', 'Birth Date', 'Birth Location',
             'Death Date', 'Death Location', 'Father', 'Father WikiTree ID', 'Father ' + self._target_field_name,
             'Father Gender',
             'Father Birth Date', 'Father Birth Location', 'Father Death Date', 'Father Death Location',
             'Mother', 'WikiTree ID.1', self._target_field_name + '.1', 'Gender.1',
             'Birth Date.1', 'Birth Location.1', 'Death Date.1', 'Death Location.1'])

        son_father_mother_sf = son_father_mother_sf.rename({'WikiTree ID.1': 'Mother WikiTree ID',
                                                            self._target_field_name + '.1': 'Mother ' + self._target_field_name,
                                                            'Gender.1': 'Mother Gender',
                                                            'Birth Date.1': 'Mother Birth Date',
                                                            'Birth Location.1': 'Mother Birth Location',
                                                            'Death Date.1': 'Mother Death Date',
                                                            'Death Location.1': 'Mother Death Location'})
        print("Mother's calculations were finished!")
        son_father_mother_sf.export_csv(self._output_directory_path + self._results_file_name, delimiter="\t")
        #self._export_to_csv(son_father_mother_sf)
        print("Done!")

    # def clean_slack_names_and_convert_to_dates_and_countries(self):
    #     wikitree_sf = self._read_csv_file("\t")
    #
    #     #cleaning names
    #     wikitree_sf['First Name'] = wikitree_sf["First Name"].apply(lambda x: clean_content(x))
    #     wikitree_sf['Father First Name'] = wikitree_sf["Father First Name"].apply(lambda x: clean_content(x))
    #     wikitree_sf['Mother First Name'] = wikitree_sf["Mother First Name"].apply(lambda x: clean_content(x))
    #
    #     wikitree_sf['Son_First_Name'] = wikitree_sf["First Name"].apply(lambda x: x.split(" "))
    #     wikitree_sf = wikitree_sf.stack("Son_First_Name", new_column_name='Son_First_Name')
    #
    #     wikitree_sf['Father_First_Name'] = wikitree_sf["Father First Name"].apply(lambda x: x.split(" "))
    #     wikitree_sf = wikitree_sf.stack("Father_First_Name", new_column_name='Father_First_Name')
    #
    #     wikitree_sf['Mother_First_Name'] = wikitree_sf["Mother First Name"].apply(lambda x: x.split(" "))
    #     wikitree_sf = wikitree_sf.stack("Mother_First_Name", new_column_name='Mother_First_Name')
    #
    #     #wikitree_sf.reset_index()
    #
    #     wikitree_sf["Birth_Country"] = \
    #         wikitree_sf["Birth Location"].apply(lambda x: get_country_by_location(x))
    #
    #     wikitree_sf["Death_Country"] = \
    #         wikitree_sf["Death Location"].apply(lambda x: get_country_by_location(x))
    #
    #     wikitree_sf["Father_Birth_Country"] = \
    #         wikitree_sf["Father Birth Location"].apply(lambda x: get_country_by_location(x))
    #
    #     wikitree_sf["Father_Death_Country"] = \
    #         wikitree_sf["Father Death Location"].apply(lambda x: get_country_by_location(x))
    #
    #     wikitree_sf["Mother_Birth_Country"] = \
    #         wikitree_sf["Mother Birth Location"].apply(lambda x: get_country_by_location(x))
    #
    #     wikitree_sf["Mother_Death_Country"] = \
    #         wikitree_sf["Mother Death Location"].apply(lambda x: get_country_by_location(x))
    #
    #
    #     wikitree_sf["Birth_Date"] = \
    #         wikitree_sf["Birth Date"].apply(lambda x: seperate_to_year_month_day(x))
    #     wikitree_sf = wikitree_sf.unpack("Birth_Date", column_name_prefix="Child_Birth_Date")
    #
    #     wikitree_sf["Death_Date"] = \
    #         wikitree_sf["Death Date"].apply(lambda x: seperate_to_year_month_day(x))
    #     wikitree_sf = wikitree_sf.unpack("Death_Date", column_name_prefix="Child_Death_Date")
    #
    #     wikitree_sf["Father_Birth_Date"] = \
    #         wikitree_sf["Father Birth Date"].apply(lambda x: seperate_to_year_month_day(x))
    #     wikitree_sf = wikitree_sf.unpack("Father_Birth_Date", column_name_prefix="Father_Birth_Date")
    #
    #     wikitree_sf["Father_Death_Date"] = \
    #         wikitree_sf["Father Death Date"].apply(lambda x: seperate_to_year_month_day(x))
    #     wikitree_sf = wikitree_sf.unpack("Father_Death_Date", column_name_prefix="Father_Death_Date")
    #
    #     wikitree_sf["Mother_Birth_Date"] = \
    #         wikitree_sf["Mother Birth Date"].apply(lambda x: seperate_to_year_month_day(x))
    #     wikitree_sf = wikitree_sf.unpack("Mother_Birth_Date", column_name_prefix="Mother_Birth_Date")
    #
    #     wikitree_sf["Mother_Death_Date"] = \
    #         wikitree_sf["Mother Death Date"].apply(lambda x: seperate_to_year_month_day(x))
    #     wikitree_sf = wikitree_sf.unpack("Mother_Death_Date", column_name_prefix="Mother_Death_Date")
    #
    #     wikitree_sf.rename({'Child_Birth_Date.0': 'Child_Birth_Year',
    #                         'Child_Birth_Date.1': 'Child_Birth_Month',
    #                         'Child_Birth_Date.2': 'Child_Birth_Day',
    #                         'Child_Death_Date.0': 'Child_Death_Year',
    #                         'Child_Death_Date.1': 'Child_Death_Month',
    #                         'Child_Death_Date.2': 'Child_Death_Day',
    #                         'Father_Birth_Date.0': 'Father_Birth_Year',
    #                         'Father_Birth_Date.1': 'Father_Birth_Month',
    #                         'Father_Birth_Date.2': 'Father_Birth_Day',
    #                         'Father_Death_Date.0': 'Father_Death_Year',
    #                         'Father_Death_Date.1': 'Father_Death_Month',
    #                         'Father_Death_Date.2': 'Father_Death_Day',
    #                         'Mother_Birth_Date.0': 'Mother_Birth_Year',
    #                         'Mother_Birth_Date.1': 'Mother_Birth_Month',
    #                         'Mother_Birth_Date.2': 'Mother_Birth_Day',
    #                         'Mother_Death_Date.0': 'Mother_Death_Year',
    #                         'Mother_Death_Date.1': 'Mother_Death_Month',
    #                         'Mother_Death_Date.2': 'Mother_Death_Day'
    #                         })
    #
    #
    #     #self._export_to_csv(son_father_mother_sf)
    #     wikitree_sf.export_csv(self._output_directory_path + self._results_file_name, delimiter="\t")
    #     print("Done!")


    def clean_slack_names_and_convert_to_dates_and_countries(self):
        wikitree_sf = self._read_csv_file("\t")

        targeted_field_name = self._target_field_name.replace(" ", "_")

        wikitree_sf['Son_' + targeted_field_name] = wikitree_sf[self._target_field_name].apply(
            lambda x: [sub_name for sub_name in x.split(" ") if len(sub_name) > 2])
        #wikitree_sf = wikitree_sf.stack('Child_' + targeted_field_name, new_column_name='Child_' + targeted_field_name)

        wikitree_sf['Father_Ancestor_' + targeted_field_name] = wikitree_sf["Father " + self._target_field_name].apply(
            lambda x: [sub_name for sub_name in x.split(" ") if len(sub_name) > 2])
        #wikitree_sf = wikitree_sf.stack('Father_' + targeted_field_name,
         #                               new_column_name='Father_' + targeted_field_name)

        wikitree_sf['Mother_Ancestor_' + targeted_field_name] = wikitree_sf["Mother " + self._target_field_name].apply(
            lambda x: [sub_name for sub_name in x.split(" ") if len(sub_name) > 2])
        #wikitree_sf = wikitree_sf.stack('Mother_' + targeted_field_name,
        #                                new_column_name='Mother_' + targeted_field_name)

        wikitree_sf['Child_' + targeted_field_name] = wikitree_sf['Son_' + targeted_field_name].apply(lambda x: [clean_content(sub_name) for sub_name in x])
        wikitree_sf = wikitree_sf.stack('Child_' + targeted_field_name, new_column_name='Child_' + targeted_field_name)

        wikitree_sf["Father_" + targeted_field_name] = wikitree_sf["Father_Ancestor_" + targeted_field_name].apply(lambda x: [clean_content(sub_name) for sub_name in x])
        wikitree_sf = wikitree_sf.stack('Father_' + targeted_field_name,
                                                                      new_column_name='Father_' + targeted_field_name)

        wikitree_sf["Mother_" + targeted_field_name] = wikitree_sf["Mother_Ancestor_" + targeted_field_name].apply(lambda x: [clean_content(sub_name) for sub_name in x])
        wikitree_sf = wikitree_sf.stack('Mother_' + targeted_field_name,
                                                                        new_column_name='Mother_' + targeted_field_name)


        wikitree_sf["Child_Birth_Country"] = \
            wikitree_sf["Birth Location"].apply(lambda x: get_country_by_location(x))

        wikitree_sf["Child_Death_Country"] = \
            wikitree_sf["Death Location"].apply(lambda x: get_country_by_location(x))

        wikitree_sf["Father_Birth_Country"] = \
            wikitree_sf["Father Birth Location"].apply(lambda x: get_country_by_location(x))

        wikitree_sf["Father_Death_Country"] = \
            wikitree_sf["Father Death Location"].apply(lambda x: get_country_by_location(x))

        wikitree_sf["Mother_Birth_Country"] = \
            wikitree_sf["Mother Birth Location"].apply(lambda x: get_country_by_location(x))

        wikitree_sf["Mother_Death_Country"] = \
            wikitree_sf["Mother Death Location"].apply(lambda x: get_country_by_location(x))


        wikitree_sf["Birth_Date"] = \
            wikitree_sf["Birth Date"].apply(lambda x: seperate_to_year_month_day(x))
        wikitree_sf = wikitree_sf.unpack("Birth_Date", column_name_prefix="Child_Birth_Date")

        wikitree_sf["Death_Date"] = \
            wikitree_sf["Death Date"].apply(lambda x: seperate_to_year_month_day(x))
        wikitree_sf = wikitree_sf.unpack("Death_Date", column_name_prefix="Child_Death_Date")

        wikitree_sf["Father_Birth_Date"] = \
            wikitree_sf["Father Birth Date"].apply(lambda x: seperate_to_year_month_day(x))
        wikitree_sf = wikitree_sf.unpack("Father_Birth_Date", column_name_prefix="Father_Birth_Date")

        wikitree_sf["Father_Death_Date"] = \
            wikitree_sf["Father Death Date"].apply(lambda x: seperate_to_year_month_day(x))
        wikitree_sf = wikitree_sf.unpack("Father_Death_Date", column_name_prefix="Father_Death_Date")

        wikitree_sf["Mother_Birth_Date"] = \
            wikitree_sf["Mother Birth Date"].apply(lambda x: seperate_to_year_month_day(x))
        wikitree_sf = wikitree_sf.unpack("Mother_Birth_Date", column_name_prefix="Mother_Birth_Date")

        wikitree_sf["Mother_Death_Date"] = \
            wikitree_sf["Mother Death Date"].apply(lambda x: seperate_to_year_month_day(x))
        wikitree_sf = wikitree_sf.unpack("Mother_Death_Date", column_name_prefix="Mother_Death_Date")

        wikitree_sf.rename({'Child_Birth_Date.0': 'Child_Birth_Year',
                            'Child_Birth_Date.1': 'Child_Birth_Month',
                            'Child_Birth_Date.2': 'Child_Birth_Day',
                            'Child_Death_Date.0': 'Child_Death_Year',
                            'Child_Death_Date.1': 'Child_Death_Month',
                            'Child_Death_Date.2': 'Child_Death_Day',
                            'Father_Birth_Date.0': 'Father_Birth_Year',
                            'Father_Birth_Date.1': 'Father_Birth_Month',
                            'Father_Birth_Date.2': 'Father_Birth_Day',
                            'Father_Death_Date.0': 'Father_Death_Year',
                            'Father_Death_Date.1': 'Father_Death_Month',
                            'Father_Death_Date.2': 'Father_Death_Day',
                            'Mother_Birth_Date.0': 'Mother_Birth_Year',
                            'Mother_Birth_Date.1': 'Mother_Birth_Month',
                            'Mother_Birth_Date.2': 'Mother_Birth_Day',
                            'Mother_Death_Date.0': 'Mother_Death_Year',
                            'Mother_Death_Date.1': 'Mother_Death_Month',
                            'Mother_Death_Date.2': 'Mother_Death_Day'
                            })


        #self._export_to_csv(son_father_mother_sf)
        wikitree_sf.export_csv(self._output_directory_path + self._results_file_name, delimiter="\t")
        print("Done!")

    def create_big_graph_GN(self):
        wikitree_sf = self._read_csv_file("\t")
        father_child_sf = wikitree_sf.select_columns(['Father_First_Name', 'Child_First_Name'])
        mother_child_sf = wikitree_sf.select_columns(['Mother_First_Name', 'Child_First_Name'])

        father_child_sf = father_child_sf.rename({'Father_First_Name': 'Ancestor_First_Name'})
        mother_child_sf = mother_child_sf.rename({'Mother_First_Name': 'Ancestor_First_Name'})

        ancestor_child_sf = father_child_sf.append(mother_child_sf)

        ancestor_child_sf = ancestor_child_sf.groupby(key_columns=['Ancestor_First_Name', 'Child_First_Name'],
            operations={'count': agg.COUNT()})

        ancestor_child_sf = ancestor_child_sf.sort('count', ascending = False)


        # child_ancestor_count_greater_than_1.export_csv(
        #     self._output_directory_path + "last_names_child_ancestor_count_greater_than_1.csv", delimiter="\t")
        ancestor_child_sf.export_csv(
            self._output_directory_path + "big_gn_graph_edges.csv", delimiter="\t")

    def calculate_measures_for_first_names(self):
        wikitree_sf = self._read_csv_file("\t")

        wikitree_sf['Edit_Distance_Son_Father'] = wikitree_sf.apply(
            lambda x: calulate_edit_distance(x["Son_First_Name"], x["Father_First_Name"]))

        wikitree_sf['Edit_Distance_Son_Mother'] = wikitree_sf.apply(
            lambda x: calulate_edit_distance(x["Son_First_Name"], x["Mother_First_Name"]))

        wikitree_sf['Soundex_Son'] = wikitree_sf.apply(
            lambda x: get_soundex(x["Son_First_Name"]))

        wikitree_sf['Soundex_Father'] = wikitree_sf.apply(
            lambda x: get_soundex(x["Father_First_Name"]))

        wikitree_sf['Soundex_Mother'] = wikitree_sf.apply(
            lambda x: get_soundex(x["Mother_First_Name"]))

        wikitree_sf['Diff_Soundex_Son_Father'] = wikitree_sf.apply(
            lambda x: get_number_of_different_characters_between_soundexes(x["Soundex_Son"],
                                                                           x["Soundex_Father"]))

        wikitree_sf['Diff_Soundex_Son_Mother'] = wikitree_sf.apply(
            lambda x: get_number_of_different_characters_between_soundexes(x["Soundex_Son"],
                                                                           x["Soundex_Mother"]))

        wikitree_sf['Is_Child_and_Father_Start_The_Same'] = wikitree_sf.apply(
            lambda x: is_two_names_start_the_same_based_on_soundex(x["Soundex_Son"], x["Soundex_Father"]))

        wikitree_sf['Is_Child_and_Mother_Start_The_Same'] = wikitree_sf.apply(
            lambda x: is_two_names_start_the_same_based_on_soundex(x["Soundex_Son"], x["Soundex_Mother"]))

        self._export_to_csv(wikitree_sf)


    def calculate_measures_for_given_names(self):
        wikitree_sf = self._read_csv_file("\t")

        targeted_field_name = self._target_field_name.replace(" ", "_")

        wikitree_sf['Edit_Distance_Child_Father'] = wikitree_sf.apply(
            lambda x: calulate_edit_distance(x["Child_" + targeted_field_name], x["Father_" + targeted_field_name]))

        wikitree_sf['Edit_Distance_Child_Mother'] = wikitree_sf.apply(
            lambda x: calulate_edit_distance(x["Child_" + targeted_field_name], x["Mother_" + targeted_field_name]))

        wikitree_sf['Soundex_Child'] = wikitree_sf.apply(
            lambda x: get_soundex(x["Child_" + targeted_field_name]))

        wikitree_sf['Soundex_Father'] = wikitree_sf.apply(
            lambda x: get_soundex(x["Father_" + targeted_field_name]))

        wikitree_sf['Soundex_Mother'] = wikitree_sf.apply(
            lambda x: get_soundex(x["Mother_" + targeted_field_name]))

        wikitree_sf['Diff_Soundex_Son_Father'] = wikitree_sf.apply(
            lambda x: get_number_of_different_characters_between_soundexes(x["Soundex_Child"],
                                                                           x["Soundex_Father"]))

        wikitree_sf['Diff_Soundex_Son_Mother'] = wikitree_sf.apply(
            lambda x: get_number_of_different_characters_between_soundexes(x["Soundex_Child"],
                                                                           x["Soundex_Mother"]))

        wikitree_sf['Is_Child_and_Father_Start_The_Same'] = wikitree_sf.apply(
            lambda x: is_two_names_start_the_same_based_on_soundex(x["Soundex_Child"], x["Soundex_Father"]))

        wikitree_sf['Is_Child_and_Mother_Start_The_Same'] = wikitree_sf.apply(
            lambda x: is_two_names_start_the_same_based_on_soundex(x["Soundex_Child"], x["Soundex_Mother"]))

        self._export_to_csv(wikitree_sf)


    def choose_edit_distance_1_to_3(self):
        wikitree_sf = self._read_csv_file("\t")

        wikitree_sf['Child_First_Name_Num_Chars'] = wikitree_sf.apply(
            lambda x: get_num_of_characters(x["Child_First_Name"]))

        wikitree_sf['Father_First_Name_Num_Chars'] = wikitree_sf.apply(
            lambda x: get_num_of_characters(x["Father_First_Name"]))

        wikitree_sf['Mother_First_Name_Num_Chars'] = wikitree_sf.apply(
            lambda x: get_num_of_characters(x["Mother_First_Name"]))

        wikitree_sf = wikitree_sf[(wikitree_sf['Child_First_Name_Num_Chars'] > 2) &
                                  (wikitree_sf['Father_First_Name_Num_Chars'] > 2) &
                                  (wikitree_sf['Mother_First_Name_Num_Chars'] > 2)]


        first_name_son_father_edit_distance_1_3_sf = wikitree_sf[
            ((wikitree_sf['Edit_Distance_Child_Father'] == 1) |
             (wikitree_sf['Edit_Distance_Child_Father'] == 2) |
             (wikitree_sf['Edit_Distance_Child_Father'] == 3))]

        first_name_son_mother_edit_distance_1_3_sf = wikitree_sf[
            ((wikitree_sf['Edit_Distance_Child_Mother'] == 1) |
             (wikitree_sf['Edit_Distance_Child_Mother'] == 2) |
             (wikitree_sf['Edit_Distance_Child_Mother'] == 3))]
             #(wikitree_sf['Is_Child_and_Mother_Start_The_Same'] == 1))]

        first_name_son_father_edit_distance_1_3_sf.export_csv(
            self._output_directory_path + "child_father_with_dates_and_measures_and_ED_1_3.csv",
            delimiter="\t")

        first_name_son_mother_edit_distance_1_3_sf.export_csv(
            self._output_directory_path + "child_mother_with_dates_and_measures_and_ED_1_3.csv",
            delimiter="\t")

        print("Done!")

    def remove_edges_with_names_lower_than_three_characters(self):
        child_father_edges_sf = SFrame.read_csv(self._input_directory_path + "first_names_child_father_group_by_ED_1_3.csv",
                                                delimiter="\t")
        child_mother_edges_sf = SFrame.read_csv(self._input_directory_path + "first_name_child_mother_group_by_ED_1_3.csv",
                                                delimiter="\t")

        child_father_edges_sf['Child_First_Name_Num_Chars'] = child_father_edges_sf.apply(
            lambda x: get_num_of_characters(x["Child_First_Name"]))

        child_father_edges_sf['Father_First_Name_Num_Chars'] = child_father_edges_sf.apply(
            lambda x: get_num_of_characters(x["Father_First_Name"]))

        child_father_edges_sf = child_father_edges_sf[(child_father_edges_sf['Child_First_Name_Num_Chars'] > 2) &
                                  (child_father_edges_sf['Father_First_Name_Num_Chars'] > 2)]


        child_mother_edges_sf['Child_First_Name_Num_Chars'] = child_mother_edges_sf.apply(
            lambda x: get_num_of_characters(x["Child_First_Name"]))

        child_mother_edges_sf['Mother_First_Name_Num_Chars'] = child_mother_edges_sf.apply(
            lambda x: get_num_of_characters(x["Mother_First_Name"]))

        child_mother_edges_sf = child_mother_edges_sf[(child_mother_edges_sf['Child_First_Name_Num_Chars'] > 2) &
                                                      (child_mother_edges_sf['Mother_First_Name_Num_Chars'] > 2)]

        child_father_edges_sf.export_csv(
            self._output_directory_path + "first_name_child_father_group_by_ED_1_3_removed_names_lower_than_three_words.csv",
            delimiter="\t")

        child_mother_edges_sf.export_csv(
            self._output_directory_path + "first_name_child_mother_group_by_ED_1_3_removed_names_lower_than_three_words.csv",
            delimiter="\t")


    def edit_distance_1_and_start_the_same(self):
        wikitree_sf = self._read_csv_file("\t")

        first_name_son_father_start_the_same_and_edit_distance_1_sf = wikitree_sf[
            ((wikitree_sf['Edit_Distance_Child_Father'] == 1) &
                          (wikitree_sf['Is_Child_and_Father_Start_The_Same'] == 1))]

        first_name_son_mother_start_the_same_and_edit_distance_1_sf = wikitree_sf[
            ((wikitree_sf['Edit_Distance_Child_Mother'] == 1) &
             (wikitree_sf['Is_Child_and_Mother_Start_The_Same'] == 1))]

        first_name_son_father_start_the_same_and_edit_distance_1_sf.export_csv(
            self._output_directory_path + "child_father_with_dates_and_measures_start_the_same_and_ED_1.csv",
            delimiter="\t")

        first_name_son_mother_start_the_same_and_edit_distance_1_sf.export_csv(
            self._output_directory_path + "child_mother_with_dates_and_measures_start_the_same_and_ED_1.csv",
            delimiter="\t")

        print("Done!")

    def remove_children_with_no_birth_date_and_location(self):
        first_name_son_father_start_the_same_and_ED_1 = SFrame.read_csv(
            #self._input_directory_path + "last_names_child_father_with_dates_and_measures_start_the_same_and_ED_1.csv",
            self._input_directory_path + "child_father_with_dates_and_measures_and_ED_1_3.csv",
            delimiter="\t")

        first_name_son_mother_start_the_same_and_ED_1 = SFrame.read_csv(
            #self._input_directory_path + "last_names_child_mother_with_dates_and_measures_start_the_same_and_ED_1.csv",
            self._input_directory_path + "child_mother_with_dates_and_measures_and_ED_1_3.csv",
            delimiter="\t")

        first_name_son_father_sorted = first_name_son_father_start_the_same_and_ED_1[
            (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Year"] != 0) &
            (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Year"] != '0') &
            (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Year"] != '') &
            (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Country"] != '') &
            (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Country"] != None)]

        first_name_son_mother_sorted = first_name_son_mother_start_the_same_and_ED_1[
            (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != 0) &
            (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != '0') &
            (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != '') &
            (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Country"] != '') &
            (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Country"] != None)]


        # first_name_son_mother_sorted = first_name_son_mother_start_the_same_and_ED_1[
        #     (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != 0) &
        #     (first_name_son_mother_start_the_same_and_ED_1["Birth_Country"] != '')]

        first_name_son_father_sorted.export_csv(
            self._output_directory_path + "first_names_child_father_with_dates_and_measures_and_ED_1_3_remove_children_no_birth_year_and_date.csv",
            delimiter="\t")

        first_name_son_mother_sorted.export_csv(
            self._output_directory_path + "first_names_child_mother_with_dates_and_measures_and_ED_1_3_remove_children_no_birth_year_and_date.csv",
            delimiter="\t")

        print("Done!")



    def first_names_edit_distance_1(self):
        wikitree_sf = self._read_csv_file("\t")

        first_name_son_father_start_the_same_and_edit_distance_1_sf = wikitree_sf[
            ((wikitree_sf['Edit_Distance_Son_Father'] == 1) &
                          (wikitree_sf['Is_Child_and_Father_Start_The_Same'] == 1))]

        first_name_son_mother_start_the_same_and_edit_distance_1_sf = wikitree_sf[
            ((wikitree_sf['Edit_Distance_Son_Mother'] == 1) &
             (wikitree_sf['Is_Child_and_Mother_Start_The_Same'] == 1))]

        first_name_son_father_start_the_same_and_edit_distance_1_sf.export_csv(
            self._output_directory_path + "first_name_son_father_with_dates_and_measures_start_the_same_and_ED_1.csv",
            delimiter="\t")

        first_name_son_mother_start_the_same_and_edit_distance_1_sf.export_csv(
            self._output_directory_path + "first_name_son_mother_with_dates_and_measures_start_the_same_and_ED_1.csv",
            delimiter="\t")

        print("Done!")


    # def remove_children_with_no_birth_date_and_location(self):
    #     first_name_son_father_start_the_same_and_ED_1 = SFrame.read_csv(
    #         self._input_directory_path + "last_names_child_father_with_dates_and_measures_start_the_same_and_ED_1.csv",
    #         delimiter="\t")
    #
    #     first_name_son_mother_start_the_same_and_ED_1 = SFrame.read_csv(
    #         self._input_directory_path + "last_names_child_mother_with_dates_and_measures_start_the_same_and_ED_1.csv",
    #         delimiter="\t")
    #
    #     first_name_son_father_sorted = first_name_son_father_start_the_same_and_ED_1[
    #         (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Year"] != 0) &
    #         (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Year"] != '0') &
    #         (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Year"] != '') &
    #         (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Country"] != '') &
    #         (first_name_son_father_start_the_same_and_ED_1["Child_Birth_Country"] != None)]
    #
    #     first_name_son_mother_sorted = first_name_son_mother_start_the_same_and_ED_1[
    #         (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != 0) &
    #         (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != '0') &
    #         (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != '') &
    #         (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Country"] != '') &
    #         (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Country"] != None)]
    #
    #
    #     # first_name_son_mother_sorted = first_name_son_mother_start_the_same_and_ED_1[
    #     #     (first_name_son_mother_start_the_same_and_ED_1["Child_Birth_Year"] != 0) &
    #     #     (first_name_son_mother_start_the_same_and_ED_1["Birth_Country"] != '')]
    #
    #     first_name_son_father_sorted.export_csv(
    #         self._output_directory_path + "last_names_son_father_with_dates_and_measures_start_the_same_and_ED_1_remove_children_no_birth_year_and_date.csv",
    #         delimiter="\t")
    #
    #     first_name_son_mother_sorted.export_csv(
    #         self._output_directory_path + "last_names_son_mother_with_dates_and_measures_start_the_same_and_ED_1_remove_children_no_birth_year_and_date.csv",
    #         delimiter="\t")
    #
    #     print("Done!")



    def group_by_for_son_father_and_son_mother(self):
        # first_name_son_father_sf = SFrame.read_csv(
        #     self._input_directory_path + "first_name_son_father_with_dates_and_measures_start_the_same_and_ED_1.csv",
        #                                delimiter="\t")
        # first_name_son_father_sf = SFrame.read_csv(
        #     self._input_directory_path + "first_names_child_father_with_dates_and_measures_start_the_same_and_ED_1_remove_children_no_birth_year_and_date.csv",
        #     delimiter="\t")
        first_name_son_father_sf = SFrame.read_csv(
            self._input_directory_path + "first_names_child_father_with_dates_and_measures_and_ED_1_3_remove_children_no_birth_year_and_date.csv",
            delimiter="\t")

        # first_name_son_mother_sf = SFrame.read_csv(
        #     self._input_directory_path + "first_name_son_mother_with_dates_and_measures_start_the_same_and_ED_1.csv",
        #     delimiter="\t")

        # first_name_son_mother_sf = SFrame.read_csv(
        #     self._input_directory_path + "first_names_child_mother_with_dates_and_measures_start_the_same_and_ED_1_remove_children_no_birth_year_and_date.csv",
        #     delimiter="\t")

        first_name_son_mother_sf = SFrame.read_csv(
            self._input_directory_path + "first_names_child_mother_with_dates_and_measures_and_ED_1_3_remove_children_no_birth_year_and_date.csv",
            delimiter="\t")

        targeted_field_name = self._target_field_name.replace(" ", "_")
        son_father_group_by_sf = first_name_son_father_sf.groupby(key_columns=['Child_'+ targeted_field_name, 'Father_' + targeted_field_name],
                                                        operations={'count': agg.COUNT()})
        son_father_group_by_sf = son_father_group_by_sf.sort(['count'], ascending=False)

        son_mother_group_by_sf = first_name_son_mother_sf.groupby(key_columns=['Child_' + targeted_field_name, 'Mother_' + targeted_field_name],
                                                                  operations={'count': agg.COUNT()})
        son_mother_group_by_sf = son_mother_group_by_sf.sort(['count'], ascending=False)

        son_father_group_by_sf.export_csv(self._output_directory_path + "first_names_child_father_group_by_ED_1_3.csv", delimiter="\t")
        son_mother_group_by_sf.export_csv(self._output_directory_path + "first_name_child_mother_group_by_ED_1_3.csv", delimiter="\t")

        # son_father_group_by_sf.export_csv(self._output_directory_path + "last_names_child_father_group_by.csv", delimiter="\t")
        # son_mother_group_by_sf.export_csv(self._output_directory_path + "last_names_child_mother_group_by.csv", delimiter="\t")
        print("Done!!!!")


    def unite_child_father_and_son_mother_ocuurances(self):
        child_father_group_by = SFrame.read_csv(
            self._input_directory_path + "first_name_child_father_group_by_ED_1_3_removed_names_lower_than_three_words.csv",
            delimiter="\t")
        child_father_group_by = child_father_group_by.remove_column("Child_First_Name_Num_Chars")
        child_father_group_by = child_father_group_by.remove_column("Father_First_Name_Num_Chars")
        # child_father_group_by = SFrame.read_csv(
        #     self._input_directory_path + "last_names_child_father_group_by.csv",
        #     delimiter="\t")

        child_mother_group_by = SFrame.read_csv(
            self._input_directory_path + "first_name_child_mother_group_by_ED_1_3_removed_names_lower_than_three_words.csv",
            delimiter="\t")
        # child_mother_group_by = SFrame.read_csv(
        #     self._input_directory_path + "last_names_child_mother_group_by.csv",
        #     delimiter="\t")

        child_mother_group_by = child_mother_group_by.remove_column("Child_First_Name_Num_Chars")
        child_mother_group_by = child_mother_group_by.remove_column("Mother_First_Name_Num_Chars")


        child_father_group_by_higher_than_1_sf = child_father_group_by[child_father_group_by["count"] > 9]
        child_mother_group_by_higher_than_1_sf = child_mother_group_by[child_mother_group_by["count"] > 9]

        targeted_field_name = self._target_field_name.replace(" ", "_")
        child_father_group_by_higher_than_1_sf.rename({'Father_' + targeted_field_name: 'Ancestor_' + targeted_field_name})

        # child_father_group_by_higher_than_1_sf.rename({'Child_First_Name': 'Child_First_Name',
        #                                                'Father_First_Name': 'Ancestor_First_Name'})

        child_mother_group_by_higher_than_1_sf.rename({'Mother_' + targeted_field_name: 'Ancestor_' + targeted_field_name})


        child_ancestor_count_greater_than_1 = child_father_group_by_higher_than_1_sf.append(child_mother_group_by_higher_than_1_sf)

        # child_ancestor_count_greater_than_1.export_csv(
        #     self._output_directory_path + "last_names_child_ancestor_count_greater_than_1.csv", delimiter="\t")
        child_ancestor_count_greater_than_1.export_csv(
            self._output_directory_path + "first_names_child_ancestor_count_greater_than_9.csv", delimiter="\t")

        child_father_group_by_higher_than_1_df = child_father_group_by_higher_than_1_sf.to_dataframe()
        child_mother_group_by_higher_than_1_df = child_mother_group_by_higher_than_1_sf.to_dataframe()

        # df = pd.concat([child_father_group_by_higher_than_1_df, child_mother_group_by_higher_than_1_df]) \
        #     .groupby(['Child_Last_Name_Current','Ancestor_Last_Name_Current'])['count'] \
        #     .sum().reset_index().sort_values(by='count', ascending=False)
        df = pd.concat([child_father_group_by_higher_than_1_df, child_mother_group_by_higher_than_1_df]) \
            .groupby(['Child_First_Name', 'Ancestor_First_Name'])['count'] \
            .sum().reset_index().sort_values(by='count', ascending=False)

        #df.to_csv(self._output_directory_path + "last_names_child_ancestor_count_greater_than_1_aggregated.csv", index=False)
        df.to_csv(self._output_directory_path + "first_names_child_ancestor_count_greater_than_9_aggregated.csv", index=False)

        print("Done!!!!")



    def export_nodes_file(self):
        node_features = []
        wikitree_sf = self._read_csv_file("\t")
        unique_names = wikitree_sf[self._target_field_name].unique()
        target_field_name = self._target_field_name
        for unique_name in unique_names:
            for node_feature in self._node_features:
                value = globals()[node_feature](target_field_name, unique_name, wikitree_sf)
                node_feature = (unique_name, node_feature, value)
                node_features.append(node_feature)

        nodes_df = pd.DataFrame(node_features, columns=['name', 'attribute_name', 'attribute_value'])
        nodes_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        pivoted_nodes_df = nodes_df.pivot(columns='attribute_name', values='attribute_value', index='name')
        pivoted_nodes_df.to_csv(self._output_directory_path + "nodes.csv")

    def export_nodes_features_file(self):
        node_features = []
        wikitree_sf = self._read_csv_file("\t")
        unique_names = wikitree_sf[self._target_field_name].unique()
        target_field_name = self._target_field_name
        for unique_name in unique_names:
            for node_feature in self._node_features:
                value = globals()[node_feature](target_field_name, unique_name, wikitree_sf)
                node_feature = (unique_name, node_feature, value)
                node_features.append(node_feature)

        nodes_df = pd.DataFrame(node_features, columns=['name', 'attribute_name', 'attribute_value'])
        nodes_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        pivoted_nodes_df = nodes_df.pivot(columns='attribute_name', values='attribute_value', index='name')
        pivoted_nodes_df.to_csv(self._output_directory_path + "nodes.csv")

    def export_nodes_info_features_file(self):
        child_ancestor_count_greater_than_1_sf = SFrame.read_csv(
            self._input_directory_path + "first_name_child_ancestor_count_greater_than_1.csv",
                                      delimiter="\t")
        # run for initilize a dict
        #alias_country_name_dict = self._convert_df_to_dict()

        child_father_mother_sf = SFrame.read_csv(
            #self._input_directory_path + "son_father_mother_last_name_dates_and_countries_with_measures.csv",
            self._input_directory_path + "son_father_mother_first_names_dates_countries_and_measures.csv",
            delimiter="\t")

        targeted_field_name = self._target_field_name.replace(" ", "_")
        ancestor_names = child_ancestor_count_greater_than_1_sf["Ancestor_" + targeted_field_name]
        child_first_names = child_ancestor_count_greater_than_1_sf["Child_" + targeted_field_name]
        all_names = child_first_names.append(ancestor_names)
        unique_names = all_names.unique()
        sf = child_father_mother_sf.filter_by(unique_names, "Child_" + targeted_field_name)

        # filtring and stay with only the people who have birth date
        sf = sf.filter_by(0, "Child_Birth_Year", exclude=True)
        # filtring again and including that has location
        sf = sf[(sf["Child_Birth_Year"] != 0)&(sf["Child_Birth_Country"] != "")]
        name_min_year_country_sf = sf.groupby("Child_" + targeted_field_name, [agg.MIN("Child_Birth_Year"), agg.SELECT_ONE("Child_Birth_Country")])

        name_min_year_country_sf['Origin_Country'] = name_min_year_country_sf.apply(
            lambda x: get_country_name_by_bing_api(x["Select One of Child_Birth_Country"]))

        name_min_year_country_sf['Earliest_Century'] = name_min_year_country_sf.apply(
            lambda x: calculate_earliest_century(int(x["Min of Child_Birth_Year"])))

        name_min_year_country_sf['Earliest_Decade'] = name_min_year_country_sf.apply(
            lambda x: calculate_earliest_decade(int(x["Min of Child_Birth_Year"])))

        name_min_year_country_sf.export_csv(self._output_directory_path + "nodes.csv")

        with open(alias_country_dict_location, 'wb') as handle:
            pickle.dump(alias_country_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def export_nodes_origin_country_and_birth_year_with_no_bing_api(self):
        # child_ancestor_count_greater_than_1_sf = SFrame.read_csv(
        #     self._input_directory_path + "first_name_child_ancestor_count_greater_than_1.csv",
        #                               delimiter="\t")
        child_ancestor_count_greater_than_1_sf = SFrame.read_csv(
            self._input_directory_path + "last_names_child_ancestor_count_greater_than_1_aggregated.csv",
            delimiter=",")
        # run for initilize a dict
        #alias_country_name_dict = self._convert_df_to_dict()

        # child_father_mother_sf = SFrame.read_csv(
        #     #self._input_directory_path + "son_father_mother_last_name_dates_and_countries_with_measures.csv",
        #     self._input_directory_path + "son_father_mother_first_names_dates_countries_and_measures.csv",
        #     delimiter="\t")
        child_father_mother_sf = SFrame.read_csv(
            #self._input_directory_path + "son_father_mother_last_name_dates_and_countries_with_measures.csv",
            self._input_directory_path + "son_father_mother_last_names_dates_and_countries.csv",
            delimiter="\t")

        targeted_field_name = self._target_field_name.replace(" ", "_")
        ancestor_names = child_ancestor_count_greater_than_1_sf["Ancestor_" + targeted_field_name]
        child_first_names = child_ancestor_count_greater_than_1_sf["Child_" + targeted_field_name]
        all_names = child_first_names.append(ancestor_names)
        unique_names = all_names.unique()
        sf = child_father_mother_sf.filter_by(unique_names, "Child_" + targeted_field_name)

        # filtring and stay with only the people who have birth date
        sf = sf.filter_by(0, "Child_Birth_Year", exclude=True)
        # filtring again and including that has location
        sf = sf[(sf["Child_Birth_Year"] != 0)&(sf["Child_Birth_Country"] != "")]
        name_min_year_country_sf = sf.groupby("Child_" + targeted_field_name, [agg.MIN("Child_Birth_Year"), agg.SELECT_ONE("Child_Birth_Country")])

        name_min_year_country_sf['Origin_Country'] = name_min_year_country_sf.apply(
            lambda x: get_country_name_by_saved_location_dictionary(x["Select One of Child_Birth_Country"]))

        name_min_year_country_sf['Earliest_Century'] = name_min_year_country_sf.apply(
            lambda x: calculate_earliest_century(int(x["Min of Child_Birth_Year"])))

        name_min_year_country_sf['Earliest_Decade'] = name_min_year_country_sf.apply(
            lambda x: calculate_earliest_decade(int(x["Min of Child_Birth_Year"])))

        name_min_year_country_sf.export_csv(self._output_directory_path + "last_names_nodes.csv")

        # with open(alias_country_dict_location, 'wb') as handle:
        #     pickle.dump(alias_country_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def _convert_df_to_dict(self):
        # read file that we already calculated and saved the locations
        df = pd.read_csv(self._input_directory_path + "first_name_node_features.csv", delimiter="\t")
        location_df = df[['Select One of Birth_Country', 'Origin_Country']]
        alias_country_name_dict = dict(list(zip(location_df['Select One of Birth_Country'], location_df['Origin_Country'])))

        full_model_file_path = self._output_directory_path + "alias_country_name_dict.pkl"
        joblib.dump(alias_country_name_dict, full_model_file_path)

        return alias_country_name_dict


    def export_nodes_year_features_file(self):
        nodes_sf = SFrame.read_csv(self._input_directory_path + "nodes.csv", delimiter="\t")

        nodes_sf['Earliest_Century'] = nodes_sf.apply(
            lambda x: calculate_earliest_century(int(x["Min of Child_Birth_Year"])))

        nodes_sf['Earliest_Decade'] = nodes_sf.apply(
            lambda x: calculate_earliest_decade(int(x["Min of Child_Birth_Year"])))

        nodes_sf.export_csv(self._output_directory_path + "node_features.csv",
                                            delimiter="\t")


    def _read_csv_file(self, delimiter="\t"):
        print("Reading CSV file ")
        begin_time = time.time()
        # for Micky's big file
        # wikitree_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter="\t")
        wikitree_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter=delimiter)
        end_time = time.time()
        run_time = end_time - begin_time
        print(run_time)
        return wikitree_sf

    def _export_to_csv(self, sf):
        print("Exporting to CSV")
        begin_time = time.time()
        sf.export_csv(self._output_directory_path + self._results_file_name, delimiter="\t")
        end_time = time.time()
        run_time = end_time - begin_time
        print(run_time)

    def evaluate_names_performance_by_name_graph(self):
        names_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        # names_df = names_df.rename(index=str, columns={"Child_Last_Name_Current": "source",
        #                                                "Ancestor_Last_Name_Current": "target",
        #                                                "count": "weight"})
        names_df = names_df.rename(index=str, columns={"Child_First_Name": "source",
                                                       "Ancestor_First_Name": "target",
                                                       "count": "weight"})
        names_graph = nx.from_pandas_edgelist(names_df, edge_attr=True)
        nodes = list(names_graph.nodes)
        results = []
        for node in nodes:

            node_soundex, node_metaphone, node_double_metaphone_primary, node_double_metaphone_secondary = \
                self._calculate_self_sound_measures(node)

            neighbors_iterator = names_graph.neighbors(node)
            neighbors = [n for n in neighbors_iterator]
            for neighbor in neighbors:
                result = self._calculate_sound_measures(node, node_soundex, node_metaphone,
                                                        node_double_metaphone_primary, node_double_metaphone_secondary,
                                                        neighbor, order=1)
                results.append(result)

                neighbors_of_neighbors_iterator = names_graph.neighbors(neighbor)
                neighbors_of_neighbors = [n for n in neighbors_of_neighbors_iterator]

                for neighbor_order_2 in neighbors_of_neighbors:
                    result = self._calculate_sound_measures(node, node_soundex, node_metaphone,
                                                            node_double_metaphone_primary, node_double_metaphone_secondary,
                                                            neighbor_order_2, order=2)
                    results.append(result)

        results_df = pd.DataFrame(results, columns=['Name', 'Neighbor', 'Order', 'Name_Soundex', 'Neighbor_Soundex',
                                                    'Diff_Soundex_Chars', 'Edit_Distance_Soundexes',
                                                    'Name_Metaphone', 'Neighbor_Methophone', 'Diff_Methophone_Chars',
                                                    'Edit_Distance_Metaphones', 'Name_Double_Metaphone_Primary',
                                                    'Neighbor_Double_Metaphone_Primary', 'Diff_Double_Methophone_Primary_Chars',
                                                    'Edit_Distance_Primary_Double_Metaphones',
                                                    'Name_Double_Metaphone_Secondary',
                                                    'Neighbor_Double_Metaphone_Secondary',
                                                    'Diff_Double_Methophone_Secondary_Chars',
                                                    'Edit_Distance_Secondary_Double_Metaphones',
                                                    'Jaro_Winkler_Sim',
                                                    'Damerau_Levenshein_Sim'])

        results_df.to_csv(self._output_directory_path + self._results_file_name)

        print("Done!!!!!!!!!")

    def count_female_male_for_each_name(self):
        names_df = pd.read_csv(self._input_directory_path + "all_first_names_based_on_child_father_and_mother.csv")
        wikitree_sf = SFrame.read_csv(self._input_directory_path + "dump_people_user_full.csv", delimiter='\t')
        unique_names = names_df['Name'].unique()
        results = []
        for i, name in enumerate(unique_names):
            print("\rFirst Name: {0} {1}/{2}".format(name, i, len(unique_names)), end='')

            chosen_name_sf = wikitree_sf[wikitree_sf["First Name"] == name]
            male_chosen_name_sf = chosen_name_sf[chosen_name_sf["Gender"] == 1]
            num_of_men = male_chosen_name_sf.num_rows()

            female_chosen_name_sf = chosen_name_sf[chosen_name_sf["Gender"] == 2]
            num_of_women = female_chosen_name_sf.num_rows()

            result = (name, num_of_men, num_of_women)
            results.append(result)

        result_df = pd.DataFrame(results, columns=['Name', 'Men', 'Woman'])
        result_df.to_csv(self._output_directory_path + "name_gender_distribution.csv")

    def label_names_to_gender(self):
        wikitree_sf = SFrame.read_csv(self._input_directory_path + "son_father_mother_first_names_dates_countries_and_measures.csv", delimiter='\t')
        short_wikitree_sf = wikitree_sf.select_columns(["Child_First_Name", "Gender"])
        names = list(wikitree_sf['Child_First_Name'])
        names = list(set(names))
        # fathers = list(wikitree_sf['Father_First_Name'])
        # mothers = list(wikitree_sf['Mother_First_Name'])
        # names = list(set(children + fathers + mothers))

        names = [name for name in names if len(name) > 2]

        results = []
        for i, name in enumerate(names):
            print("\rFirst Name: {0} {1}/{2}".format(name, i, len(names)), end='')

            chosen_name_sf = short_wikitree_sf[short_wikitree_sf["Child_First_Name"] == name]
            row = chosen_name_sf.head(1)
            if row.num_rows() > 0:
                gender = row["Gender"][0]
                result = (name, gender)
                results.append(result)


        result_df = pd.DataFrame(results, columns=['Name', 'Gender'])
        result_df = result_df.sort_values(by=['Name'])
        result_df.to_csv(self._output_directory_path + "name_gender_distribution.csv")

    def evaluate_names_performance_for_all_names(self):
        names_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter='\t')

        targeted_field_name = self._target_field_name.replace(" ", "_")
        first_names = names_sf["Child_" + targeted_field_name].unique()
        first_names = [first_name for first_name in first_names if len(first_name) > 2]
        first_names = sorted(first_names)

        results = []
        for i, first_name in enumerate(first_names):
            print("\rFirst Name: {0} {1}/{2}".format(first_name, i, len(first_names)), end='')
            node_soundex, node_metaphone, node_double_metaphone_primary, node_double_metaphone_secondary, \
            name_nysiis, name_match_rating_codex = self._calculate_self_sound_measures(first_name)

            result = (first_name, node_soundex, node_metaphone, node_double_metaphone_primary,
                      node_double_metaphone_secondary, name_nysiis, name_match_rating_codex)

            results.append(result)


        results_df = pd.DataFrame(results, columns=['Name', 'Name_Soundex', 'Name_Metaphone',
                                                    'Name_Double_Metaphone_Primary',
                                                    'Name_Double_Metaphone_Secondary',
                                                    'Nysiis', 'Match_Rating_Codex'])

        results_df.to_csv(self._output_directory_path + self._results_file_name)

        print("Done!!!!!!!!!")

    def create_distinct_list_of_names(self):
        names_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter='\t')



    def _calculate_self_sound_measures(self, name):
        name_soundex = get_soundex(name)
        name_metaphone = get_phonetics_metaphone(name)
        name_double_metaphone = get_phonetics_double_metaphone(name)
        name_double_metaphone_primary = name_double_metaphone[0]
        name_double_metaphone_secondary = name_double_metaphone[1]
        name_nysiis = get_nysiis(name)
        name_match_rating_codex = get_match_rating_codex(name)


        return name_soundex, name_metaphone, name_double_metaphone_primary, \
               name_double_metaphone_secondary, name_nysiis, name_match_rating_codex

    def _calculate_sound_measures(self, node, node_soundex, node_metaphone, node_double_metaphone_primary,
                                  node_double_metaphone_secondary, neighbor, order):

        neighbor_soundex, neighbor_metaphone, neighbor_double_metaphone_primary, neighbor_double_metaphone_secondary = \
            self._calculate_self_sound_measures(neighbor)
        diff_num_chars_soundex = get_number_of_different_characters_between_soundexes(node_soundex,
                                                                                      neighbor_soundex)
        diff_edit_distance_soundex = calulate_edit_distance(node_soundex, neighbor_soundex)

        diff_num_chars_metaphone = get_number_of_different_characters_between_soundexes(node_metaphone,
                                                                                        neighbor_metaphone)
        diff_edit_distance_metaphone = calulate_edit_distance(node_metaphone, neighbor_metaphone)

        diff_num_chars_double_metaphone_primary = get_number_of_different_characters_between_soundexes(node_double_metaphone_primary,
                                                                                                       neighbor_double_metaphone_primary)
        diff_num_chars_double_metaphone_secondary = get_number_of_different_characters_between_soundexes(
            node_double_metaphone_secondary,
            neighbor_double_metaphone_secondary)

        diff_edit_distance_double_metaphone_primary = calulate_edit_distance(node_double_metaphone_primary,
                                                                             neighbor_double_metaphone_primary)
        diff_edit_distance_double_metaphone_secondary = calulate_edit_distance(node_double_metaphone_secondary,
                                                                             neighbor_double_metaphone_secondary)

        jaro_winkler_sim = jaro_winkler_name_similarity(node, neighbor)
        damerau_levenshtein_sim = damerau_levenshtein_similarity(node, neighbor)

        result = (node, neighbor, order, node_soundex, neighbor_soundex, diff_num_chars_soundex, diff_edit_distance_soundex,
                  node_metaphone, neighbor_metaphone, diff_num_chars_metaphone, diff_edit_distance_metaphone,
                  node_double_metaphone_primary, neighbor_double_metaphone_primary, diff_num_chars_double_metaphone_primary,
                  diff_edit_distance_double_metaphone_primary, node_double_metaphone_secondary,
                  neighbor_double_metaphone_secondary, diff_num_chars_double_metaphone_secondary,
                  diff_edit_distance_double_metaphone_secondary, jaro_winkler_sim, damerau_levenshtein_sim)

        return result

    def create_all_distinct_names_list_based_on_son_father_and_mother(self):
        wikitree_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter='\t')
        targeted_field_name = self._target_field_name.replace(" ", "_")
        child_first_names_series = wikitree_sf["Child_{0}".format(targeted_field_name)]
        child_first_names = list(child_first_names_series)

        father_first_names_series = wikitree_sf["Father_{0}".format(targeted_field_name)]
        father_first_names = list(father_first_names_series)

        mother_first_names_series = wikitree_sf["Mother_{0}".format(targeted_field_name)]
        mother_first_names = list(mother_first_names_series)

        first_names = child_first_names + father_first_names + mother_first_names
        first_names = list(set(first_names))
        first_names = [name for name in first_names if len(name) > 2]
        first_names = sorted(first_names)

        first_names_df = pd.DataFrame(first_names, columns=['Name'])
        first_names_df.to_csv(self._output_directory_path + self._results_file_name)


    def get_all_distinct_cleaned_names_list(self):
        wikitree_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter='\t')

        short_wikitree_sf = wikitree_sf.select_columns([self._target_field_name])

        short_wikitree_sf = short_wikitree_sf[(short_wikitree_sf[self._target_field_name] != None) &
                                  (short_wikitree_sf[self._target_field_name] != '') &
                                  (short_wikitree_sf[self._target_field_name] != 'Unknown') &
                                  (short_wikitree_sf[self._target_field_name] != 'Anonymous')]


        targeted_field_name = self._target_file_name
        short_wikitree_sf['Person ' + targeted_field_name] = short_wikitree_sf[self._target_field_name].apply(
            lambda x: [sub_name for sub_name in x.split(" ") if len(sub_name) > 2])

        short_wikitree_sf['Person_' + targeted_field_name] = short_wikitree_sf['Person ' + targeted_field_name].apply(
            lambda x: [clean_content(sub_name) for sub_name in x])
        short_wikitree_sf = short_wikitree_sf.stack('Person_' + targeted_field_name, new_column_name='Person_' + targeted_field_name)

        first_names_series = short_wikitree_sf['Person_' + targeted_field_name]
        first_names = list(first_names_series)

        first_names = list(set(first_names))
        first_names = [name for name in first_names if name is not None and len(name) > 2]
        first_names = sorted(first_names)

        first_names_df = pd.DataFrame(first_names, columns=['Name'])
        first_names_df.to_csv(self._output_directory_path + self._results_file_name, index=False)


def earliest_year(target_field, name, wikitree_sf):
    child_selected_name_sf = wikitree_sf[wikitree_sf[target_field] == name]
    earliest_year_child = child_selected_name_sf['Child_Birth_Year'].min()
    return earliest_year_child

def earliest_century(target_field, name, wikitree_sf):
    earlier_year = earliest_year(target_field, name, wikitree_sf)
    return calculate_earliest_century(earlier_year)

def earliest_decade(target_field, name, wikitree_sf):
    earlier_year = earliest_year(target_field, name, wikitree_sf)
    century = earlier_year / 100
    decimal_year = earlier_year - (century * 100)
    decade = decimal_year / 10
    return decade

def calculate_earliest_century(earlier_year):
    century = earlier_year // 100 + 1
    return century

def calculate_earliest_decade(earlier_year):
    century = earlier_year / 100
    decimal_year = earlier_year - (century * 100)
    decade = decimal_year / 10
    return decade


def origin_country(target_field, name, wikitree_sf):
    earlier_year = earliest_year(target_field, name, wikitree_sf)

    earliest_year_name_sf = wikitree_sf[
        (wikitree_sf[target_field] == name) & (wikitree_sf['Child_Birth_Year'] == earlier_year)]
    birth_country = earliest_year_name_sf['Birth_Country'][0]
    return birth_country

def get_phonetics_nysiss(name):
    if name is not None and name is not 'None' and name is not '':
        print(name)
        result = phonetics.nysiis(name)
        return result
    return ''

def get_phonetics_metaphone(name):
    if name is not None and name is not 'None' and name is not '':
        # name = unicode(name)
        result = phonetics.metaphone(name)
        return result
    return ''

def get_phonetics_double_metaphone(name):
    if name is not None and name is not 'None' and name is not '':
        # name = unicode(name)
        result = phonetics.dmetaphone(name)
        return result
    return ''

names_edit_distance_dict = {}

def calulate_edit_distance(name1, name2):
    if not name1 or not name2:
        return -1

    name1 = name1.lower()
    name2 = name2.lower()

    key = name1 + " -> " + name2

    # name1 = name1.replace("-", "")
    # name2 = name2.replace("-", "")
    #
    # name1 = name1.replace(" ", "")
    # name2 = name2.replace(" ", "")

    if key not in names_edit_distance_dict:
        edit_dist = editdistance.eval(name1, name2)
        names_edit_distance_dict[key] = edit_dist

    return names_edit_distance_dict[key]

name_soundex_dict = {}

def get_soundex(name):
    if name is not None and name is not 'None' and name is not '':
        print(name)
        name = str(name)


        if name not in name_soundex_dict:
            soundex_result = jellyfish.soundex(name)
            name_soundex_dict[name] = soundex_result

        return name_soundex_dict[name]
    return ''

def soundex_name_similarity(name1, name2):
    if not name1 or not name2:
        return -1

    name1_soundex_result = get_soundex(name1)
    name2_soundex_result = get_soundex(name2)

    if name1_soundex_result == name2_soundex_result:
        return 1
    return 0

def get_number_of_different_characters_between_soundexes(soundex1, soundex2):
    if soundex1 is not '' and soundex2 is not '':
        diff = 0
        num_of_chars_soundex_1 = len(soundex1)
        num_of_chars_soundex_2 = len(soundex2)

        if num_of_chars_soundex_1 != num_of_chars_soundex_2:
            return -1
        i = 0
        while i < num_of_chars_soundex_1:
            # print("Soundex1:{0}, Soundex2:{1}".format(soundex1, soundex2))
            soundex1_char = soundex1[i]
            soundex2_char = soundex2[i]
            if soundex1_char != soundex2_char:
                diff += 1
            i += 1
        return diff
    return -1

# def is_two_names_start_the_same_based_on_soundex(soundex1, soundex2):
#     if soundex1 is not None and soundex2 is not None \
#             and soundex1 is not -1 and soundex2 is not -1 \
#             and soundex1 is not 'None' and soundex2 is not 'None' \
#             and not soundex1 and not soundex2:
#         #print("Soundex1: {0}".format(soundex1))
#         soundex1_char = soundex1[0]
#
#         #print("Soundex2: {0}".format(soundex2))
#         soundex2_char = soundex2[0]
#         if soundex1_char == soundex2_char:
#             return 1
#     return 0

def remove_second_names(name):
    print(name)
    names = name.split(" ")
    first_name = names[0]
    return first_name

def is_two_names_start_the_same_based_on_soundex(soundex1, soundex2):
    if soundex1 is not '' and soundex2 is not '':
        # print("Soundex1: {0}".format(soundex1))
        soundex1_char = soundex1[0]

        # print("Soundex2: {0}".format(soundex2))
        soundex2_char = soundex2[0]
        if soundex1_char == soundex2_char:
            return 1
        return 0
    return -1

def difference_name_similarity(name1, name2):
    if not name1 or not name2:
        return -1

    name1_soundex_result = get_soundex(name1)
    name2_soundex_result = get_soundex(name2)

    name1_soundex_result_set = set(name1_soundex_result)
    name2_soundex_result_set = set(name2_soundex_result)

    common_characters_set = name1_soundex_result_set & name2_soundex_result_set
    num_common_characters = len(list(common_characters_set))

    # Each soundex return 4 characters. The results find how many characters in common
    result = num_common_characters / float(4)
    return result

def lcs_name_similarity(name1, name2):
    if not name1 or not name2:
        return -1

    name1 = name1.lower()
    name2 = name2.lower()

    name1_elements = name1.split(" ")
    name2_elements = name2.split(" ")
    tuples = list(itertools.product(name1_elements, name2_elements))
    longest_sub_strings = []
    for tuple in tuples:
        sub_name1 = tuple[0]
        sub_name2 = tuple[1]
        sequences = list(py_common_subseq.find_common_subsequences(sub_name1, sub_name2))
        sequences.sort(key=len, reverse=True)
        longest_str = sequences[0]
        # longest_str is not ""
        if longest_str:
            longest_sub_strings.append(longest_str)
    average_sub_strings_length = (len(name1) + len(name2)) / float(2)
    sum_sub_strings_length = sum(map(len, longest_sub_strings))
    result = sum_sub_strings_length / average_sub_strings_length
    return result

def damerau_levenshtein_similarity(name1, name2):
    if not name1 or not name2:
        return -1

    name1 = str(name1)
    name2 = str(name2)

    damerau_levenshtein_distance = jellyfish.damerau_levenshtein_distance(name1, name2)
    return damerau_levenshtein_distance

def jaro_winkler_name_similarity(name1, name2):
    if not name1 or not name2:
        return -1
    name1 = str(name1)
    name2 = str(name2)
    jaro_winkler_distance = jellyfish.jaro_winkler(name1, name2)
    return jaro_winkler_distance

def get_nysiis(name):
    name = str(name)
    nysiis = jellyfish.nysiis(name)
    return nysiis

def get_match_rating_codex(name):
    name = str(name)
    match_rating_codex = jellyfish.match_rating_codex(name)
    return match_rating_codex

def convert_epoch_time_to_date(time):
    str_time = str(time)
    if len(str_time) == 4:
        return time
    date_time = datetime.datetime.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
    return date_time

def find_diff_between_names(son_name, ancestor_name):
    son_name = son_name.lower()
    ancestor_name = ancestor_name.lower()

    son_name_length = len(son_name)
    ancestor_name_length = len(ancestor_name)
    if son_name_length > ancestor_name_length:
        ancestor_characters = list(ancestor_name)
        for character in ancestor_characters:
            son_name = son_name.replace(character, '', 1)
            additional_character = find_additional_character(son_name, ancestor_name)
            result = "son_added_{0}".format(additional_character)
            return result
    elif son_name_length < ancestor_name_length:
        additional_character = find_additional_character(ancestor_name, son_name)
        result = "son_removed_{0}".format(additional_character)
        return result
    else:
        for i in range(len(son_name)):
            if son_name[i] != ancestor_name[i]:
                result = "ancestor_{0}-son_{1}".format(ancestor_name[i], son_name[i])
                return result

# Example: long_name = "Stumpf", short_name="Stump" will return "f"
def find_additional_character(long_name, short_name):
    short_name_characters = list(short_name)
    for character in short_name_characters:
        long_name = long_name.replace(character, '', 1)
    # will be a letter that left
    return long_name

def identify_diff_between_son_and_ancestor_with_edit_distance_2(son_name, ancestor_name):
    # son_name = "Wambolt"
    # ancestor_name = "Womboldt"

    # ancestor_name = "Wambolt"
    # son_name = "Womboldt"

    # son_name = "Mosley"
    # ancestor_name = "Mosely"

    # ancestor_name = "Mosley"
    # son_name = "Mosely"

    # son_name = "Stumpff"
    # ancestor_name = "Stump"

    # ancestor_name = "Stumpff"
    # son_name = "Stump"

    # son_name = "Ffstump"
    # ancestor_name = "Stump"

    # ancestor_name = "Ffstump"
    # son_name = "Stump"

    # son_name = "Stffump"
    # ancestor_name = "Stump"

    # son_name = "Young"
    # ancestor_name = "Jung"

    # ancestor_name = "Young"
    # son_name = "Jung"
    #
    # son_name = "Youmh"
    # ancestor_name = "Young"
    #
    # ancestor_name = "Youmh"
    # son_name = "Young"

    son_name = son_name.lower()
    ancestor_name = ancestor_name.lower()

    son_num_of_chars = len(son_name)
    ancestor_num_of_chars = len(ancestor_name)

    i = 0
    j = 0
    changes = []

    if ancestor_name in son_name:
        rest = son_name.replace(ancestor_name, '')
        for char in rest:
            result = "son_added_{0}".format(char)
            change = (son_name, ancestor_name, result)
            changes.append(change)
    elif son_name in ancestor_name:
        rest = ancestor_name.replace(son_name, '')
        for char in rest:
            result = "son_removed_{0}".format(char)
            change = (son_name, ancestor_name, result)
            changes.append(change)

    while i < son_num_of_chars and j < ancestor_num_of_chars and len(changes) < 2:
        son_char = son_name[i]
        ancestor_char = ancestor_name[j]
        if son_char == ancestor_char:
            i += 1
            j += 1

        else:
            if i + 1 < son_num_of_chars and j + 1 < ancestor_num_of_chars and son_name[i + 1] == ancestor_name[
                j + 1]:
                result = "ancestor_{0}-son_{1}".format(ancestor_char, son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)
                i += 1
                j += 1
            elif ancestor_num_of_chars > son_num_of_chars and j + 1 < ancestor_num_of_chars \
                    and son_name[i] == ancestor_name[j + 1]:
                result = "son_removed_{0}".format(ancestor_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)
                j += 1
            elif ancestor_num_of_chars < son_num_of_chars and i + 1 < son_num_of_chars \
                    and son_name[i + 1] == ancestor_name[j]:
                result = "son_added_{0}".format(son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                i += 1

            elif i + 2 < son_num_of_chars and son_name[i + 2] == ancestor_char:

                result = "son_added_{0}".format(son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                i + 1

            elif i + 2 < ancestor_num_of_chars and ancestor_name[i + 2] == son_char:
                result = "son_removed_{0}".format(ancestor_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                j += 1

            elif i + 2 < son_num_of_chars and j + 1 < ancestor_num_of_chars and son_name[i + 2] == ancestor_name[
                j + 1]:
                result = "son_added_{0}".format(son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                i += 1

            elif i + 1 < son_num_of_chars and j + 2 < ancestor_num_of_chars and son_name[i + 1] == ancestor_name[
                j + 2]:
                result = "son_removed_{0}".format(ancestor_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                j += 1

            else:
                result = "ancestor_{0}-son_{1}".format(ancestor_char, son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                if i + 2 < son_num_of_chars and j + 2 < ancestor_num_of_chars and son_name[i + 2] != ancestor_name[
                    j + 2]:
                    if ancestor_num_of_chars > son_num_of_chars:
                        j += 1
                    elif ancestor_num_of_chars < son_num_of_chars:
                        i += 1
                else:
                    i += 1
                    j += 1

    return changes

def identify_diff_between_son_and_ancestor_with_edit_distance_2_one_string(son_name, ancestor_name):
    # son_name = "Wambolt"
    # ancestor_name = "Womboldt"

    # ancestor_name = "Wambolt"
    # son_name = "Womboldt"

    # son_name = "Mosley"
    # ancestor_name = "Mosely"

    # ancestor_name = "Mosley"
    # son_name = "Mosely"

    # son_name = "Stumpff"
    # ancestor_name = "Stump"

    # ancestor_name = "Stumpff"
    # son_name = "Stump"

    # son_name = "Ffstump"
    # ancestor_name = "Stump"

    # ancestor_name = "Ffstump"
    # son_name = "Stump"

    # son_name = "Stffump"
    # ancestor_name = "Stump"

    # son_name = "Young"
    # ancestor_name = "Jung"

    # ancestor_name = "Young"
    # son_name = "Jung"
    #
    # son_name = "Youmh"
    # ancestor_name = "Young"
    #
    # ancestor_name = "Youmh"
    # son_name = "Young"

    son_name = son_name.lower()
    ancestor_name = ancestor_name.lower()

    son_name = son_name.replace(" ", "")
    ancestor_name = ancestor_name.replace(" ", "")

    son_name = son_name.replace("-", "")
    ancestor_name = ancestor_name.replace("-", "")

    son_num_of_chars = len(son_name)
    ancestor_num_of_chars = len(ancestor_name)

    i = 0
    j = 0
    changes = []

    if ancestor_name in son_name:
        rest = son_name.replace(ancestor_name, '')
        for char in rest:
            result = "son_added_{0}".format(char)
            change = (son_name, ancestor_name, result)
            changes.append(change)
    elif son_name in ancestor_name:
        rest = ancestor_name.replace(son_name, '')
        for char in rest:
            result = "son_removed_{0}".format(char)
            change = (son_name, ancestor_name, result)
            changes.append(change)

    while i < son_num_of_chars and j < ancestor_num_of_chars and len(changes) < 2:
        son_char = son_name[i]
        ancestor_char = ancestor_name[j]
        if son_char == ancestor_char:
            i += 1
            j += 1

        else:
            if i + 1 < son_num_of_chars and j + 1 < ancestor_num_of_chars and son_name[i + 1] == ancestor_name[
                j + 1]:
                result = "ancestor_{0}-son_{1}".format(ancestor_char, son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)
                i += 1
                j += 1
            elif ancestor_num_of_chars > son_num_of_chars and j + 1 < ancestor_num_of_chars \
                    and son_name[i] == ancestor_name[j + 1]:
                result = "son_removed_{0}".format(ancestor_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)
                j += 1
            elif ancestor_num_of_chars < son_num_of_chars and i + 1 < son_num_of_chars \
                    and son_name[i + 1] == ancestor_name[j]:
                result = "son_added_{0}".format(son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                i += 1

            elif i + 2 < son_num_of_chars and son_name[i + 2] == ancestor_char:

                result = "son_added_{0}".format(son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                i + 1

            elif i + 2 < ancestor_num_of_chars and ancestor_name[i + 2] == son_char:
                result = "son_removed_{0}".format(ancestor_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                j += 1

            elif i + 2 < son_num_of_chars and j + 1 < ancestor_num_of_chars and son_name[i + 2] == ancestor_name[
                j + 1]:
                result = "son_added_{0}".format(son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                i += 1

            elif i + 1 < son_num_of_chars and j + 2 < ancestor_num_of_chars and son_name[i + 1] == ancestor_name[
                j + 2]:
                result = "son_removed_{0}".format(ancestor_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                j += 1

            else:
                result = "ancestor_{0}-son_{1}".format(ancestor_char, son_char)
                change = (son_name, ancestor_name, result)
                changes.append(change)

                if i + 2 < son_num_of_chars and j + 2 < ancestor_num_of_chars and son_name[i + 2] != ancestor_name[
                    j + 2]:
                    if ancestor_num_of_chars > son_num_of_chars:
                        j += 1
                    elif ancestor_num_of_chars < son_num_of_chars:
                        i += 1
                else:
                    i += 1
                    j += 1

    final_result = ""
    for change in changes:
        final_result += change[2] + ","
    final_result = final_result[:-1]
    return final_result

def cascade_names(name1, name2):
    result = -1
    if name1 is not None and name2 is not None:
        result = "{0}_{1}".format(name1, name2)
    return result

name_cleaned_name_dict = {}
def clean_content(name):
    if name not in name_cleaned_name_dict:
        regex = re.compile('[^a-zA-Z]')
        # First parameter is the replacement, second parameter is your input string
        cleaned_name = regex.sub('', name)

        cleaned_name = cleaned_name.title()
        name_cleaned_name_dict[name] = cleaned_name

    return name_cleaned_name_dict[name]

def convert_name_to_array(name):
    names = []
    sub_names = name.split(" ")
    for sub_name in sub_names:
        if len(sub_name) > 1:
            names.append(sub_name)
    return names

def seperate_to_year_month_day(time):
    time_str = str(time)
    return time_str[:4], time_str[4:6], time_str[6:8]

us_states = set(["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
                 "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
                 "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
                 "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
                 "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
                 "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
                 "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
                 "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
                 "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
                 "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                 "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                 "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                 "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
                 "United States", 'U.S.A', 'U.S.', 'U.S', 'New England', "Conn", "United States of America",
                 "Republic of Vermont", "Massachusetts Bay", "New Hampton. New Hampshire", "USA"
                 ])

canada_states = set(['Alberta', 'British Columbia', 'Manitoba', 'New Brunswick', 'Newfoundland and Labrador',
                     'Northwest Territories', 'Nova Scotia', 'Nova Scotia Colony', 'Nunavut', 'Ontario',
                     'Prince Edward Island', 'Newfoundland',
                     'Quebec', 'Saskatchewan', 'Yukon',
                     'AB', 'BC', 'MB', 'NB', 'NL', 'NT', 'NS', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT', 'Canada'
                     ])

def get_country_by_location(location):
    locations = location.split(",")
    last_location = locations[-1].strip()
    if last_location in us_states:
        return "USA"
    if last_location in canada_states:
        return "Canada"
    return last_location


def earliest_year(target_field, name, wikitree_sf):
    child_selected_name_sf = wikitree_sf[wikitree_sf[target_field] == name]
    earliest_year_child = child_selected_name_sf['Child_Birth_Year'].min()
    return earliest_year_child

def earliest_century(target_field, name, wikitree_sf):
    earlier_year = earliest_year(target_field, name, wikitree_sf)
    century = earlier_year // 100 + 1
    return century

def earliest_decade(target_field, name, wikitree_sf):
    earlier_year = earliest_year(target_field, name, wikitree_sf)
    century = earlier_year / 100
    decimal_year = earlier_year - (century * 100)
    decade = decimal_year / 10
    return decade

def origin_country(target_field, name, wikitree_sf):
    earlier_year = earliest_year(target_field, name, wikitree_sf)

    earliest_year_name_sf = wikitree_sf[(wikitree_sf[target_field] == name) & (wikitree_sf['Child_Birth_Year'] == earlier_year)]
    birth_country = earliest_year_name_sf['Birth_Country'][0]
    return birth_country

def get_num_of_characters(x):
    return len(x)


#alias_country_dict_location = "D:/somweb/Wikitree_Names_Project/software/bad_actors/data/input/WikiTreeAnalyzer/alias_country_name_dict.pkl"
alias_country_dict_location = "D:/somweb/Wikitree_Names_Project/software/bad_actors/data/input/WikiTreeAnalyzer/alias_country_name_dict.pickle"
# with open(alias_country_dict_location, 'rb') as handle:
#     alias_country_name_dict = pickle.load(handle)
alias_country_name_dict = {}
geolocator = Bing(api_key="AkWYDYbcbWYvSEW0RiZXPKENb3E4-tpnk2Iek0jIL6iqnG9u707QUkqEoG5Z6Gqa")
def get_country_name_by_bing_api(parsed_location):
    if parsed_location not in alias_country_name_dict:
        try:
            location = geolocator.geocode(parsed_location)
            if location is None:
                return ""
            address = location.address
            locations = address.split(",")
            country_name = locations[-1]
            alias_country_name_dict[parsed_location] = country_name
            return country_name

        except GeocoderTimedOut as e:
            msg = e.message
            print(msg)
            with open(alias_country_dict_location, 'wb') as handle:
                pickle.dump(alias_country_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Dictionary was saved! ")
            count_down_time(900)
            location = geolocator.geocode(parsed_location)
            if location is None:
                return ""
            address = location.address
            locations = address.split(",")
            country_name = locations[-1]
            alias_country_name_dict[parsed_location] = country_name
            return country_name

        except GeocoderQueryError as e:
            msg = e.message
            print(msg)
            with open(alias_country_dict_location, 'wb') as handle:
                pickle.dump(alias_country_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Dictionary was saved! ")
            return "Bad_Request"

        except GeocoderQuotaExceeded as e:
            msg = e.message
            print(msg)
            with open(alias_country_dict_location, 'wb') as handle:
                pickle.dump(alias_country_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Dictionary was saved! ")
            count_down_time(2000)
            return "Bad_Request"


alias_country_dict_loc = "D:/somweb/WikiTree_Family_Names/software/bad_actors/data/input/WikiTreeAnalyzer/alias_country_name_dict.pickle"
with open(alias_country_dict_location, 'rb') as handle:
     alias_country_dict = pickle.load(handle)
def get_country_name_by_saved_location_dictionary(parsed_location):
    if parsed_location not in alias_country_dict:
        return parsed_location
    return alias_country_dict[parsed_location]



def count_down_time(seconds_to_wait):
    for i in range(seconds_to_wait, 0, -1):
        time.sleep(1)
        msg = "\rCount Down {0}".format(str(i))
        print(msg, end="")
        # sys.stdout.flush()

def get_country_by_api(parsed_location):
    country_name = find_whether_name_is_north_america_countries(parsed_location)
    if country_name != "":
        return country_name
    country_name = find_whether_is_it_country(parsed_location)
    if country_name != "":
        return country_name
    else:
        url = "https://www.meteoblue.com/en/server/search/query3?query={0}".format(parsed_location)
        try:
            response = urllib.request.urlopen(url)
            response_obj_str = response.read()
            response_obj = json.loads(response_obj_str)
            num_of_results = response_obj["count"]
            if num_of_results > 0:
                #results = response_obj["results"]
                country = response_obj["results"][0]["country"]
                return country
            return ""
        except urllib.error.HTTPError as e:
            return ""

# get name and return the name whether it is a country else return ""
def get_country_name(name):
    url = "https://restcountries.eu/rest/v2/name/{0}".format(name)
    response = urllib.request.urlopen(url)
    response_obj_str = response.read()
    x = 3

def find_whether_name_is_north_america_countries(name):
    if name in us_states:
        return "USA"
    elif name in canada_states:
        return "Canada"
    else:
        return ""

alternative_name_country_name_dict = {}
def find_whether_is_it_country(name):

    if name in alternative_name_country_name_dict:
        country_name = alternative_name_country_name_dict[name]
        return country_name
    else:
        url = "https://restcountries.eu/rest/v2/name/{0}".format(name)
        try:
            response = urllib.request.urlopen(url)
            response_obj_str = response.read()
            response_obj = json.loads(response_obj_str)
            response_obj = response_obj[0]

            country_name = response_obj["name"]
            alternative_name_country_name_dict[country_name] = country_name

            translations_dict = response_obj["translations"]
            translations = list(translations_dict.values())
            for translation in translations:
                alternative_name_country_name_dict[translation] = country_name

            altrenatives = response_obj["altSpellings"]
            for altrenative in altrenatives:
                alternative_name_country_name_dict[altrenative] = country_name

            return country_name
        except urllib.error.HTTPError as e:
            return ""
