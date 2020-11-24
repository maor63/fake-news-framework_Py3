
# Client ID
# API Key
# Client Secret


# viqYC6x5dh_qADSrKLaI7Q
# 'v_oJy7EDqV6WbKQXM0pgI9_xCcRR0BKOEhWp8RPPeIC9RDOuUnL8lwxmYMEEfOK0dVLVX0YNaeb1_8REz2RX9nRqP_Is5uhWPi9-DJe5prJFNtNfIt_eVW0H8A6VWnYx'
# qD41hVAYUnBIJFlgx0mrKCqIRV9PfNd7W4BEhX0oRN8KK2WLjBOysgz8qd8Dr0it

# eZfJjshi_UUduREwNUrikw
# SliMFBK-RLQ5V-Tg-Y4aCP3fC9Rsnr6CneXstvQZqUQX_gbPix69cj0q8lwBkJEV0h4Cvxmxq-JrDKtdbiuqDfLvPpysNYakDOu0jZBnrbRKEGaSvxqwf63j8y2JWnYx
# uJmG7ssgdfq8B0tfMc2eLa9vWT1EqkoOK56QEf8PvlpgMisR1P5pr1gQZJNllyaX

# C26tHBEXyLLaaSemB1d3uw
# SUcbmL06SH7urNFEN3hdMs16TU8tDifpylv3jiIoXJB9vpQnfyLf_7MqTP6Tr9DQdXSSW0LyAeBzMoXMSY_xzKzyRn3v2meVjbT62TUxIcWGb526A5y-6fNAxsuKWnYx
# m0lKMvh9x8RX7nXJTl02jDKTvKsrDmcIqTINS1YmUQn7E8ReVYSslXc97SPN5qZ7

# P4UY0ehAEpKX9rJY6doq0w
# 3Y3eRYMtwt-StJd7MGGdXOenLZEFTbFFvN2KyPNP9g-MUddr02vTamRP52dEGjkL-7oLznTxwEG2IW6Jwo-zhMDx0Yba-qQAWIudmYwWM_Wm8QVuz-KhepzkVoeNWnYx'
# 2n0jiW7Ibc3dOKYq9ZpvcN02tEGm5EAwavRPvd9ca8lk11xjIB3IdihX1E2Pp5BA

# ALTCFHjOSENZZi__5_MhVA
# 'ZlBau_g7tgO1ZO6odmrMs44K5JsDI1-A8w_2iPd8DjOtNXC2pE0Tm4hTiZrimCMBV2ro5MmhcO3GrHb26zymKsuJFjkiGeD5U8M097HBgBzAZ8ZzI770MDfnXA2VWnYx'
# 7YdvV4JsxEUlvoIIuvhNpxO7xxbt41U1ymPD2oONjPURdNplHLa1CDMeSuzmZNYb



"""
    Copyright (c) 2013, Los Alamos National Security, LLC
    All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Los Alamos National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import logging

import requests

AUTOCOMPLETE_API_URL = 'https://api.yelp.com/v3/autocomplete'
BUSINESS_API_URL = 'https://api.yelp.com/v3/businesses/{}'
BUSINESS_MATCH_API_URL = 'https://api.yelp.com/v3/businesses/matches/{}'
EVENT_LOOKUP_API_URL = 'https://api.yelp.com/v3/events/{}'
EVENT_SEARCH_API_URL = 'https://api.yelp.com/v3/events'
FEATURED_EVENT_API_URL = 'https://api.yelp.com/v3/events/featured'
PHONE_SEARCH_API_URL = 'https://api.yelp.com/v3/businesses/search/phone'
REVIEWS_API_URL = 'https://api.yelp.com/v3/businesses/{}/reviews'
SEARCH_API_URL = 'https://api.yelp.com/v3/businesses/search'
TRANSACTION_SEARCH_API_URL = 'https://api.yelp.com/v3/transactions/{}/search'


class YelpAPI(object):
    """
        This class implements the complete Yelp Fusion API. It offers access to the following APIs:

            * Autocomplete API - https://www.yelp.com/developers/documentation/v3/autocomplete
            * Business API - https://www.yelp.com/developers/documentation/v3/business
            * Business Match API - https://www.yelp.com/developers/documentation/v3/business_match
            * Event Lookup API - https://www.yelp.com/developers/documentation/v3/event
            * Event Search API - https://www.yelp.com/developers/documentation/v3/event_search
            * Featured Event API - https://www.yelp.com/developers/documentation/v3/featured_event
            * Phone Search API - https://www.yelp.com/developers/documentation/v3/business_search_phone
            * Reviews API - https://www.yelp.com/developers/documentation/v3/business_reviews
            * Search API - https://www.yelp.com/developers/documentation/v3/business_search
            * Transaction Search API - https://www.yelp.com/developers/documentation/v3/transactions_search
        It is simple and completely extensible since it dynamically takes arguments. This will allow it to continue
        working even if Yelp changes the spec. The only thing that should cause this to break is if Yelp changes the URL
        scheme.
        The structure of each method is quite simple. Some parameters help form the request URL (e.g., a ID of some
        sort), and other parameters are a part of the JSON request. Parameters that are a part of the URL are
        explicitly asked for as a part of the method definition. Parameters that are a part of the JSON request
        are passed via `**kwargs`. Some parameters are required, and others are optional. To avoid unnecessarily using
        precious API calls, each method explicitly checks for parameters that are required in order for the query to
        succeed before issuing the call.
    """

    class YelpAPIError(Exception):

        """
            This class is used for all API errors. Currently, there is no master list of all possible errors, but
            there is an open issue on this: https://github.com/Yelp/yelp-fusion/issues/95.
        """
        pass

    def __init__(self):
        """
            Instantiate a YelpAPI object. An API key from Yelp is required. 
        """
        self._init()

    def _init(self):
        self._keys = ('rHQ0xLx57JK1j6DdkF4VHBXZ4zvq8maTRRiIG71XC3ai32tDx-ehecPnPbGkekwIW5mftw7HnI6dHJhFX_SgHWKzOcLtFfJ8MD3eSmCo9n2awczucTFpSG6TyJqWWnYx',
                      'niuNMYYhBGbwYqEXTJEMYjEQBjyuaIYN5smaUwYdN1XYmFnUXd5EAxFnF1qFPj3O1UU-XHCDvs0infe7W_bavXnmR7uoxF8wq4rm4gBXXrMgolUQERwlzHr1Q5uWWnYx'
                    'v_oJy7EDqV6WbKQXM0pgI9_xCcRR0BKOEhWp8RPPeIC9RDOuUnL8lwxmYMEEfOK0dVLVX0YNaeb1_8REz2RX9nRqP_Is5uhWPi9-DJe5prJFNtNfIt_eVW0H8A6VWnYx',
                      'SliMFBK-RLQ5V-Tg-Y4aCP3fC9Rsnr6CneXstvQZqUQX_gbPix69cj0q8lwBkJEV0h4Cvxmxq-JrDKtdbiuqDfLvPpysNYakDOu0jZBnrbRKEGaSvxqwf63j8y2JWnYx',
                      'SUcbmL06SH7urNFEN3hdMs16TU8tDifpylv3jiIoXJB9vpQnfyLf_7MqTP6Tr9DQdXSSW0LyAeBzMoXMSY_xzKzyRn3v2meVjbT62TUxIcWGb526A5y-6fNAxsuKWnYx',
                      '3Y3eRYMtwt-StJd7MGGdXOenLZEFTbFFvN2KyPNP9g-MUddr02vTamRP52dEGjkL-7oLznTxwEG2IW6Jwo-zhMDx0Yba-qQAWIudmYwWM_Wm8QVuz-KhepzkVoeNWnYx',
                      'ZlBau_g7tgO1ZO6odmrMs44K5JsDI1-A8w_2iPd8DjOtNXC2pE0Tm4hTiZrimCMBV2ro5MmhcO3GrHb26zymKsuJFjkiGeD5U8M097HBgBzAZ8ZzI770MDfnXA2VWnYx')
        self._key_num = -1
        self._api_key = 0
        self._change_key()
        self._yelp_session = requests.Session()
        self._headers = {'Authorization': 'Bearer {}'.format(self._api_key)}

    def autocomplete_query(self, **kwargs):
        """
            Query the Yelp Autocomplete API.

            documentation: https://www.yelp.com/developers/documentation/v3/autocomplete
            required parameters:
                * text - search text
        """
        if not kwargs.get('text'):
            raise ValueError('Valid text (parameter "text") must be provided.')

        return self._query(AUTOCOMPLETE_API_URL, **kwargs)

    def business_query(self, id, **kwargs):
        """
            Query the Yelp Business API.

            documentation: https://www.yelp.com/developers/documentation/v3/business
            required parameters:
                * id - business ID
        """
        if not id:
            raise ValueError('A valid business ID (parameter "id") must be provided.')

        return self._query(BUSINESS_API_URL.format(id), **kwargs)

    def business_match_query(self, match_type='best', **kwargs):
        """
            Query the Yelp Business Match API.

            documentation: https://www.yelp.com/developers/documentation/v3/business_match
            required parameters:
                * name - business name
                * city
                * state
                * country
        """
        if not kwargs.get('name'):
            raise ValueError('Valid business name (parameter "name") must be provided.')

        if not kwargs.get('city'):
            raise ValueError('Valid city (parameter "city") must be provided.')

        if not kwargs.get('state'):
            raise ValueError('Valid state (parameter "state") must be provided.')

        if not kwargs.get('country'):
            raise ValueError('Valid country (parameter "country") must be provided.')

        if match_type not in {'best', 'lookup'}:
            raise ValueError('Valid match type (parameter "match_type") must be provided. Accepted values: "best" or '
                             '"lookup".')

        return self._query(BUSINESS_MATCH_API_URL.format(match_type), **kwargs)

    def event_lookup_query(self, id, **kwargs):
        """
            Query the Yelp Event Lookup API.

            documentation: https://www.yelp.com/developers/documentation/v3/event
            required parameters:
                * id - event ID
        """
        if not id:
            raise ValueError('A valid event ID (parameter "id") must be provided.')

        return self._query(EVENT_LOOKUP_API_URL.format(id), **kwargs)

    def event_search_query(self, **kwargs):
        """
            Query the Yelp Event Search API.

            documentation: https://www.yelp.com/developers/documentation/v3/event_search
        """
        return self._query(EVENT_SEARCH_API_URL, **kwargs)

    def featured_event_query(self, **kwargs):
        """
            Query the Yelp Featured Event API.
            documentation: https://www.yelp.com/developers/documentation/v3/featured_event
            required parameters:
                * one of either:
                    * location - text specifying a location to search for
                    * latitude and longitude
        """
        if not kwargs.get('location') and (not kwargs.get('latitude') or not kwargs.get('longitude')):
            raise ValueError('A valid location (parameter "location") or latitude/longitude combination '
                             '(parameters "latitude" and "longitude") must be provided.')

        return self._query(FEATURED_EVENT_API_URL, **kwargs)

    def phone_search_query(self, **kwargs):
        """
            Query the Yelp Phone Search API.

            documentation: https://www.yelp.com/developers/documentation/v3/business_search_phone
            required parameters:
                * phone - phone number
        """
        if not kwargs.get('phone'):
            raise ValueError('A valid phone number (parameter "phone") must be provided.')

        return self._query(PHONE_SEARCH_API_URL, **kwargs)

    def reviews_query(self, id, **kwargs):
        """
            Query the Yelp Reviews API.

            documentation: https://www.yelp.com/developers/documentation/v3/business_reviews
            required parameters:
                * id - business ID
        """
        if not id:
            raise ValueError('A valid business ID (parameter "id") must be provided.')

        return self._query(REVIEWS_API_URL.format(id), **kwargs)

    def search_query(self, **kwargs):
        """
            Query the Yelp Search API.

            documentation: https://www.yelp.com/developers/documentation/v3/business_search
            required parameters:
                * one of either:
                    * location - text specifying a location to search for
                    * latitude and longitude
        """
        if not kwargs.get('location') and (not kwargs.get('latitude') or not kwargs.get('longitude')):
            raise ValueError('A valid location (parameter "location") or latitude/longitude combination '
                             '(parameters "latitude" and "longitude") must be provided.')

        return self._query(SEARCH_API_URL, **kwargs)

    def transaction_search_query(self, transaction_type, **kwargs):
        """
            Query the Yelp Transaction Search API.

            documentation: https://www.yelp.com/developers/documentation/v3/transactions_search

            required parameters:
                * transaction_type - transaction type
                * one of either:
                    * location - text specifying a location to search for
                    * latitude and longitude
        """
        if not transaction_type:
            raise ValueError('A valid transaction type (parameter "transaction_type") must be provided.')

        if not kwargs.get('location') and (not kwargs.get('latitude') or not kwargs.get('longitude')):
            raise ValueError('A valid location (parameter "location") or latitude/longitude combination '
                             '(parameters "latitude" and "longitude") must be provided.')

        return self._query(TRANSACTION_SEARCH_API_URL.format(transaction_type), **kwargs)

    @staticmethod
    def _get_clean_parameters(kwargs):
        """
            Clean the parameters by filtering out any parameters that have a None value.
        """
        return dict((k, v) for k, v in list(kwargs.items()) if v is not None)

    def _query(self, url, **kwargs):
        """
            All query methods have the same logic, so don't repeat it! Query the URL, parse the response as JSON,
            and check for errors. If all goes well, return the parsed JSON.
        """
        parameters = YelpAPI._get_clean_parameters(kwargs)
        response = self._yelp_session.get(url, headers=self._headers, params=parameters)
        response_json = response.json()  # shouldn't happen, but this will raise a ValueError if the response isn't JSON

        # Yelp can return one of many different API errors, so check for one of them.
        # The Yelp Fusion API does not yet have a complete list of errors, but this is on the TODO list; see
        # https://github.com/Yelp/yelp-fusion/issues/95 for more info.
        # if 'error' in response_json:
        #     raise YelpAPI.YelpAPIError('{}: {}'.format(response_json['error']['code'],
        #                                                response_json['error']['description']))

        # we got a good response, so return

        if 'error' in response_json and response_json['error']['code'] == 'ACCESS_LIMIT_REACHED':
            self._change_key()
            response_json= self._query(url,**kwargs)

        if 'error' in response_json and response_json['error']['code'] == 'UNAUTHORIZED_ACCESS_TOKEN':
            self._change_key()
            response_json = self._query(url, **kwargs)

        if 'error' in response_json and response_json['error']['code'] == 'TOKEN_INVALID':
            self._change_key()
            response_json = self._query(url, **kwargs)


        return response_json

    def _change_key(self):
        self._key_num=self._key_num+1
        key_index = (self._key_num % len(self._keys))
        logging.info("changed key")
        self._api_key = self._keys[key_index]
        self._headers = {'Authorization': 'Bearer {}'.format(self._api_key)}
        return self._api_key

    def get_restaurants_id(self, name, longitude, latitude, radius):
        try:
            rest = self.get_restaurant(name, longitude, latitude, radius)
            if rest is None:
                return None
            return rest['id']
        except:
            return None

    def get_restaurant(self, restaurant_name, longitude, latitude, radius):
        results = self.search_query(longitude=longitude, latitude=latitude, term=restaurant_name, radius=radius)
        try:
            if results is None or results['total']==0 or len(results['businesses']) == 0:
                return None
            return results['businesses'][0]
        except Exception as e:
            logging.info("caught exception : with restaurant: "+str(restaurant_name).encode('ascii', 'ignore').decode('ascii')+" "+str(longitude)+" , "+str(latitude))
            return None


    def get_restaurant_by_general_location(self, name, location):
        try:
            restarts = self.search_query(term=name, location=location)
            if restarts['total'] == 0:
                return None
            return restarts['businesses'][0]
        except:
            logging.info("problem with <name> <location> " +name+ " "+location)
            return None



    def get_restaurant_reviews(self, restaurant):
        rest_id = restaurant['id']
        reviews = self.reviews_query(rest_id)
        return reviews