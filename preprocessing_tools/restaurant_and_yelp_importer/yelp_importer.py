
#Written by Lior Bass 2/18/2018
#api keys in yelpapi!!!




import json
import logging

import datetime

from DB.schema_definition import Post
from configuration.config_class import getConfig
from .yelpapi import YelpAPI


class Yelp_Importer():
    def __init__(self, db):
        self._db = db
        self._yelp_api = YelpAPI()
        self._radius = 500
        self._config_parser = getConfig()
        self._reviews_json = self._config_parser.eval(self.__class__.__name__, "reviews_json_path")
        self._business_json = self._config_parser.eval(self.__class__.__name__, "business_json_path")
        self._ids_map_json = self._config_parser.eval(self.__class__.__name__, "ids_map_json")
        self._business_json_id_to_yelp_adi_id_dict = {}
        self._skipped_known_business = 0
        self._counter = 0
        self._total_counter =0
        self._unsaved_ids_mapping_dict = {}

    def setUp(self):
        pass

    def execute(self, window_start=None):
        self._restaurant_adi_id_dict = self._db.get_resturant_api_id_to_resturant_dict()
        self._load_ids_map()
        # self._parse_businesses()
        logging.info("started parsing reviews-posts")
        self._parse_reviews_and_add_to_db()


    def _parse_reviews_and_add_to_db(self):
        self._posts = []
        counter =0
        with open(self._reviews_json, 'r') as reviews_file:
            for line in reviews_file:
                self.parse_review(line)
                counter += 1
                if len(self._posts) % 50 == 1:
                    self.add_posts_to_db(counter)
            self.add_posts_to_db(counter)

    def _parse_businesses(self):
        self._adi_business_id_to_reviews_dict = {}
        print("")
        with open(self._business_json, 'r') as buisnesses_file:
            item_counter = 0
            for line in buisnesses_file:
                item_counter += 1
                if item_counter % 200 == 0:
                    # logging.info("processed " + str(item_counter) + " to business dict ")
                    print ("\r" + str(datetime.datetime.now().time()) + ":  processed " + str(item_counter) + " to business dict \n" +"skipped " + str(self._skipped_known_business) + " known businiess", end="")

                self.parse_business(line)
            self._save_ids_map_to_json()

    def parse_business(self, line):

        line_dictionary = json.loads(line)
        json_id = str(line_dictionary['business_id']).encode('ascii', 'ignore').decode('ascii')
        if line_dictionary is None:
            return None

        if json_id in self._business_json_id_to_yelp_adi_id_dict or json_id in self._unsaved_ids_mapping_dict:
            self._skipped_known_business += 1
            return None
        try:
            api_restaurant = self._yelp_api.get_restaurant(line_dictionary['name'], line_dictionary['longitude'],
                                             line_dictionary['latitude'], self._radius)
        except:
            logging.info("skipped line-yelp_api_error")
            return None
        if api_restaurant is None:
            return None
        api_id = str(api_restaurant['id']).encode('ascii', 'ignore').decode('ascii')

        if json_id not in self._business_json_id_to_yelp_adi_id_dict:
            if json_id not in self._unsaved_ids_mapping_dict:
                self._unsaved_ids_mapping_dict[json_id] = api_id
                self._business_json_id_to_yelp_adi_id_dict[json_id] = api_id

        self._save_ids_map_to_json()
        # if api_id in self._restaurant_adi_id_dict:
        #     self._save_ids_map_to_json(json_id,api_id)



    def parse_review(self, line):
        review = json.loads(line)
        json_id = review['business_id']
        json_id=str(json_id).encode('ascii', 'ignore').decode('ascii')

        if json_id in self._business_json_id_to_yelp_adi_id_dict:
            api_id = self._business_json_id_to_yelp_adi_id_dict[json_id]
            self._total_counter+=1
            if self._total_counter%500 ==0:
                logging.info("checked "+str(self._total_counter)+" rests, found  "+str(self._counter))
            if api_id in self._restaurant_adi_id_dict:
                restaurant = self._restaurant_adi_id_dict[api_id]
                post = self.add_review_to_restorunt(review, api_id, json_id)
                self._total_counter +=1
                logging.info("Added review "+str(self._counter)+"!")
                self._posts.append(post)

    def add_posts_to_db(self, counter):
        logging.info(str(counter) + " adding posts to db")
        self._db.addPosts(self._posts)
        self._db.commit()
        self._posts = []

    def add_review_to_restorunt(self,review, api_id, json_id):
        p = Post()
        p.author_guid = api_id
        p.author= json_id
        p.domain = 'Restaurant'
        p.content = review['text']
        p.created_at = review['date']
        p.favorite_count = review['useful']
        p.post_id=review['review_id']
        return p

    def setUp(self):
        pass
    def is_well_defined(self):
        return True

    def _save_ids_map_to_json(self):
        with open(self._ids_map_json, 'a+') as ids_dict_writer:
            # logging.info("Adding ids to json-api id mapper currently: " + str(len(self._unsaved_ids_mapping_dict))+ " added")
            for json_id, api_id in self._unsaved_ids_mapping_dict.items():
                ids_dict_writer.write(json_id+' '+api_id+"\n")
            self._unsaved_ids_mapping_dict= {}


    def _load_ids_map(self):
        try:
            logging.info("Reading json to api id map json file")
            with open(self._ids_map_json,'r') as ids_dict_writer:
                for line in ids_dict_writer:
                    splitted = line.split(' ')
                    json_id = splitted[0]
                    api_id = splitted[1].replace('\n','')
                    self._business_json_id_to_yelp_adi_id_dict[json_id]=api_id
            # logging.info("Read "+str(len(self._business_json_id_to_yelp_adi_id_dict))+" known ids")

        except:
            logging.info("Creating json to api id map json file")
            with open(self._ids_map_json,'wb') as ids_dict_writer:
                self._business_json_id_to_yelp_adi_id_dict={}
