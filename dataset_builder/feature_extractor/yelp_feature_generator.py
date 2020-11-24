#Written by Lior Bass 2/27/2018
import logging

from commons import commons
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.restaurant_and_yelp_importer.yelpapi import YelpAPI


class Yelp_Feature_Generator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._yelp_api = YelpAPI()
        sf_center_location =  (37.787925, -122.407515) # union square
        cg_center_location = (41.892406, -87.632629) #river north
        self._center_places = [sf_center_location,cg_center_location]
        self._api_restaurnt_dict = {}
    def cleanUp(self):
        pass
#region api features
    def distance_from_center_point(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            restaurant = kwargs['author']
            location = eval(restaurant.geo_enabled)
            min_dist = 100
            for center in self._center_places:
                distance = abs(commons.distance_calculator(center,location))
                min_dist = min(min_dist,distance)
            return min_dist
        else:
            raise Exception('Author object was not passed as parameter')

    def price_level(self, **kwargs):
        api_restaurant = self._get_restaurant(**kwargs)
        return self._get_info_from_resturant_dict(api_restaurant, 'price')

    def review_count(self, **kwargs):
        api_restaurant = self._get_restaurant(**kwargs)
        return self._get_info_from_resturant_dict(api_restaurant,'review_count')

    def number_of_categories(self, **kwargs):
        api_restaurant = self._get_restaurant(**kwargs)
        return len(api_restaurant['categories'])

    def reviews_rating(self, **kwargs):
        api_restaurant = self._get_restaurant(**kwargs)
        return self._get_info_from_resturant_dict(api_restaurant,'rating')

#endregion

#region private helpers
    def _get_info_from_resturant_dict(self, restaurant_dict, field):
        try:
            return restaurant_dict[field]
        except:
            if 'id' not in restaurant_dict:
                logging.info('no id in dict, here is what there is- '+str(restaurant_dict))
            logging.info("skipped line: " + str(restaurant_dict['id']) + " field: " + field)
            return -1

    def _get_restaurant(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            restaurant = kwargs['author']
            api_restaurant = self._get_api_restaurant(restaurant)
            return api_restaurant
        else:
            raise Exception('Author object was not passed as parameter')
    def _price_to_int(self, price):
        return len(price)

    def _get_api_restaurant(self, restaurant):
        try:
            if restaurant.author_guid in self._api_restaurnt_dict:
                return self._api_restaurnt_dict[restaurant.author_guid]
            else:
                lat, int = self._extact_coardinates(restaurant)
                api_result = self._yelp_api.get_restaurant(restaurant_name= restaurant.name, longitude=int, latitude=lat, radius=100)
                self._api_restaurnt_dict[restaurant.author_guid] = api_result
                return api_result
        except:
            logging.info("Problem with getting restaurant by id: "+str(restaurant.author_guid))

    def _extact_coardinates(self, coordinates_text):
        try:
            cor = coordinates_text.geo_enabled.replace('(', "")
            cor = cor.replace(')', "")
            cor = cor.split(', ')
            lat = float(cor[0])
            long = float(cor[1])
            if lat==0 or int==0:
                return None, None
            return lat, int
        except:
            return None, None
    #endregion