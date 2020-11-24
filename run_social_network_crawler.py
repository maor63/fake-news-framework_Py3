#
# Created by Aviad on 13-May-16 9:46 PM.
#
import logging
from logging import config

from DB.schema_definition import DB
from Twitter_API.twitter_api_requester import TwitterApiRequester
from configuration.config_class import getConfig
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api


if __name__ == '__main__':
    #config_parser = Configuration.get_config_parser()
    config_parser = getConfig()
    logging.config.fileConfig(config_parser.get("Logger", "logger_conf_file"))
    logging.info("Start program...")
    print("Start program...")

    social_network_crawler = Twitter_Rest_Api()

    #twitter_rest_api.crawl_followers()


    targeted_twitter_author_ids = []
    targeted_twitter_author_ids.append(targeted_twitter_author_id)
    are_user_ids = True
    social_network_crawler.crawl_followers_by_twitter_author_ids(targeted_twitter_author_ids, bad_actor_type,
                                                                 are_user_ids)
    # twitter_rest_api.crawl_


    '''

    db = DB()
    db.setUp()

    logging.info("Creating TwitterApiRequester")
    print("Creating TwitterApiRequester")
    twitter_api_requester = TwitterApiRequester()

    total_followers_ids = []
    total_followers = []
    detected_screen_name = config_parser.get("", "logger_conf_file")

    followers_ids_counter = 0

    followers_ids = twitter_api_requester.get_follower_ids_by_screen_name(detected_screen_name)
    total_followers_ids = list(set(total_followers_ids + followers_ids))
    followers_ids_counter += 1

    for follower_id in total_followers_ids:
        if followers_ids_counter >= 15:
            break
        follower_ids = twitter_api_requester.get_follower_ids_by_user_id(follower_id)
        total_followers_ids = list(set(total_followers_ids + follower_ids))
        followers_ids_counter += 1

    for follower_id in total_followers_ids:
        follower = twitter_api_requester.get_user_by_user_id(follower_id)

        total_followers.append(follower)

    db.add_json_authors(total_followers)
    '''
