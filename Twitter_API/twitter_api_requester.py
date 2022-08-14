# Created by aviade
# Time: 29/03/2016 16:53

#from vendors.twitter import Api
import twitter
from commons.commons import *


class TwitterApiRequester:
    def __init__(self, app_number):

        # TAringthon's app (Aviad)
        self._consumer_key_1 = "YewkYt0uOr8XLHEKeoCm5IljS"
        self._consumer_secret_1 = "qWcYeoJ1ebUScGhW9V8yseJFbaGa7wt3eaI0nceM6XHbtwqvmd"
        self._access_token_key_1 = "714699220352155648-3QXFPHQC7yEjK3T9BQuEIJ7Rny8I3q3"
        self._access_token_secret_1 = "UroyUvZVfl3guYdFKnkqdkNytkUhIKwLcw46kKHTfFQwY"
        self._user_id_1 = 714699220352155648
        self._screen_name_1 = "TAringthon"

        # meggiewill5's app (Jorge)
        self._consumer_key_2 = "zvFH3ykfSbl12MPlZsXgP4raJ"
        self._consumer_secret_2 = "IOJBcDDCxpQrK0lyfgiZkrdaOzAFsC6ZZfxqLCTVKpkHrWFhq3"
        self._access_token_key_2 = "714702531482488835-GVrddJKuyHGcBVvhfem2DF7HFPuCLMB"
        self._access_token_secret_2 = "48GdxHe3wb8GfkXdLFDkWKuhjgH2iZ5zAgGYJaxErqeXc"
        self._user_id_2 = 714702531482488835
        self._screen_name_2 = "meggiewill5"

        # LeviAvavilevi's app (Maor)
        self._consumer_key_3 = "rpprJ6rRHHjZezq9DyhjmcIoY"
        self._consumer_secret_3 = "bRCzR2osZdnv9J4mcKLaNIWXAl8WmuYUIkkFxyS430llG0bhGC"
        self._access_token_key_3 = "714707633354227712-FvOBFQs4OIsjjunNnfcuZ7DAVSeMWZt"
        self._access_token_secret_3 = "jzzGlk4uOaNjpCxBFIHQv9o28cxee7gQNoaCG1DGKlBl2"
        self._user_id_3 = 714707633354227712
        self._screen_name_3 = "LeviAvavilevi"

        #Intelici
        self._consumer_key_4 =  'ywfcmWZk9wVIdlnJ81q1cJdHx'
        self._consumer_secret_4 = 'Koo4gMez3mclCEXczbchDGN1pQvBcCMnahQsYWP8b3Xo1omj3a'
        self._access_token_key_4 = '879963143434362880-imeJhOUiRX9i0TX2mxmghkY9R23vSSb'
        self._access_token_secret_4 = 'uJ5jATuLuNRxjfgSmCiCieLg8iTWvQpsNTsvdk8ogQeZ5'

        print("-----Choose app ------")
        consumer_key, consumer_secret, access_token_key, access_token_secret = self._choose_app(app_number)
        self.create_twitter_api(consumer_key, consumer_secret, access_token_key, access_token_secret)



    def create_twitter_api(self, consumer_key, consumer_secret, access_token_key, access_token_secret):
        # self.api = Api(consumer_key=consumer_key,
        #                        consumer_secret=consumer_secret,
        #                        access_token_key=access_token_key,
        #                        access_token_secret=access_token_secret)

        self.api = twitter.Api(consumer_key=consumer_key,
                               consumer_secret=consumer_secret,
                               access_token_key=access_token_key,
                               access_token_secret=access_token_secret,
                               sleep_on_rate_limit=True,
                               timeout=1000)

        print("The twitter.Api object created")

    def _choose_app(self, app_number):
        if app_number == 1:
            consumer_key = self._consumer_key_1
            consumer_secret = self._consumer_secret_1
            access_token_key = self._access_token_key_1
            access_token_secret = self._access_token_secret_1
        elif app_number == 2:
            consumer_key = self._consumer_key_2
            consumer_secret = self._consumer_secret_2
            access_token_key = self._access_token_key_2
            access_token_secret = self._access_token_secret_2
        elif app_number == 3:
            consumer_key = self._consumer_key_3
            consumer_secret = self._consumer_secret_3
            access_token_key = self._access_token_key_3
            access_token_secret = self._access_token_secret_3
        elif app_number == 4:
            consumer_key = self._consumer_key_4
            consumer_secret = self._consumer_secret_4
            access_token_key = self._access_token_key_4
            access_token_secret = self._access_token_secret_4

        print(("The chosen app number is: " + str(app_number)))

        return (consumer_key, consumer_secret, access_token_key, access_token_secret)

    def verify_credentials(self):
        logging.info("----- api.VerifyCredentials() -------")
        print("----- api.VerifyCredentials() -------")

        authenticated_user = self.api.VerifyCredentials()

        logging.info("The authenticated user is: " + authenticated_user.screen_name)
        print(("The authenticated user is: " + authenticated_user.screen_name))
        logging.info(str(authenticated_user))
        print((str(authenticated_user)))

        return authenticated_user

    def get_authenticated_user_id(self):
        return self.authenticated_user.id

    def get_timeline_by_user_id(self, user_id):
        logging.info("get_timeline_by_user_id for user_id: " + str(user_id))
        statuses = self.api.GetUserTimeline(user_id=user_id, count=300)
        logging.info("Number of retrieved statuses is: " + str(len(statuses)))
        #self.print_list(statuses)
        return statuses

    def get_timeline_by_screen_name(self, screen_name):
        statuses = self.api.GetUserTimeline(None, screen_name)
        self.print_list(statuses)

    def print_list(self, items):
        list_for_print = "[" + "".join([str(item) + "," for item in items])
        list_for_print = list_for_print[:-1] # remove the last unnecessary last ","
        print(list_for_print + "]")

    def get_friends(self):
        friends = self.api.GetFriends()
        self.print_list(friends)
        return friends
        #print([user.name for user in users])

    def post_status_message(self, message):
        status = self.api.PostUpdate(message)
        print(status)
        return status

    def get_followers(self, user_id):
        followers = self.api.GetFollowers(user_id)
        self.print_list(followers)
        return followers
        #print "".join([str(follower) for follower in followers])

    def get_follower_ids_by_user_id(self, user_id):
        print(('--- get_follower_ids_by_user_id: ' + str(user_id)))

        #follower_ids = self.api.GetFollowerIDs(user_id=user_id, total_count=None)
        #follower_ids, cursor = self.api.GetFollowerIDs(user_id=user_id)
        follower_ids = self.api.GetFollowerIDs(user_id=user_id, total_count=10000)

        print(("Number of retrieved followers ids is: " + str(len(follower_ids))))
        return follower_ids

    def get_sleep_time_for_follower_ids(self):
        sec = self.api.GetSleepTime('/followers/ids')
        return sec

    def get_friend_ids_by_user_id(self, user_id):
        friend_ids = self.api.GetFriendIDs(user_id=user_id, total_count=5000)
        # print(friend_ids)
        print(("Number of retrieved friends ids is: " + str(len(friend_ids))))
        return friend_ids


    def get_follower_ids_by_screen_name(self, screen_name):
        follower_ids = self.api.GetFollowerIDs(screen_name=screen_name)
        print(("Number of followers ids is :" + str(len(follower_ids))))
        print(follower_ids)
        return follower_ids


    def get_retweeter_ids_by_status_id(self, status_id):
        sec_to_sleep = self.get_sleep_time_for_get_retweeter_ids_request()
        if sec_to_sleep > 0:
            time.sleep(sec_to_sleep + 10)
        retweeters = self.api.GetRetweeters(status_id, count=100)
        return retweeters

    def get_retweets_by_status_id(self, status_id):
        retweets = self.api.GetRetweets(status_id, count=None)
        # self.print_list(retweets)
        return retweets

    def get_retweets_of_me(self):
        my_retweeters = self.api.GetRetweetsOfMe()
        self.print_list(my_retweeters)
        return my_retweeters

    def get_status(self, id):
        status = self.api.GetStatus(id)
        print(status)
        return status

    def get_timeline(self, author_name, maximal_tweets_count_in_timeline):
        timeline = self.api.GetUserTimeline(screen_name=author_name, count=maximal_tweets_count_in_timeline)
        return timeline

    def get_favorites(self):
        self.authenticated_user = self.verify_credentials()
        favorites = self.api.GetFavorites(self.authenticated_user.id)
        self.print_list(favorites)
        return favorites

    def get_user_retweets(self, user_id):
        self.api.GetUserRetweets()

    def get_user_search_by_term(self, term):
        print(("------Get user search by term: " + term))
        users_search = self.api.GetUsersSearch(term)
        self.print_list(users_search)
        return users_search

    def get_tweets_by_term(self, term, result_type):
        print(("------get_tweets_by_term: " + term))
        tweets = self.api.GetSearch(term, count=100, result_type=result_type)
        self.print_list(tweets)
        return tweets

    def get_tweet_by_post_id(self, post_id):
        return self.get_tweets_by_post_ids([post_id])[0]

    def get_tweets_by_post_ids(self, post_ids):
        return self.api.GetStatuses(post_ids)

    def get_user_by_screen_name(self, screen_name):
        print(("------Get user by screen name: " + screen_name))
        twitter_user = self.api.GetUser(None, screen_name)
        print(twitter_user)
        return twitter_user

    def get_user_by_user_id(self, user_id):
        print("---------get_user_by_user_id------------")
        twitter_user = self.api.GetUser(user_id)
        print(twitter_user)
        return twitter_user

    def get_users_by_ids(self, ids):
        print("---------get_users_by_ids------------")
        users = self.api.UsersLookup(ids)
        print(("Num of retrieved twitter users is: " + str(len(users))))
        logging.info("Num of retrieved twitter users is: " + str(len(users)))
        return users

    def get_users_by_screen_names(self, screen_names):
        print("---------get_users_by_screen_names------------")
        users = self.api.UsersLookup(screen_name=screen_names)
        print(("Num of retrieved twitter users is: " + str(len(users))))
        logging.info("Num of retrieved twitter users is: " + str(len(users)))
        return users

    def get_sleep_time_for_get_users_request(self):
        print("---GetSleepTime /users/lookup ---")
        logging.info("---GetSleepTime /users/lookup ---")

        seconds_to_wait_object = self.api.CheckRateLimit('/users/lookup')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        print(("Seconds to wait for CheckRateLimit('/users/lookup') is: " + str(seconds_to_wait)))
        return seconds_to_wait

    def get_sleep_time_for_get_tweets_by_tweet_ids_request(self):
        print("---GetSleepTime /statuses/lookup ---")
        logging.info("---GetSleepTime statuses/lookup ---")

        seconds_to_wait_object = self.api.CheckRateLimit('/statuses/lookup')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        print(("Seconds to wait for GetSleepTime('/statuses/lookup') is: " + str(seconds_to_wait)))
        return seconds_to_wait

    def get_sleep_time_for_timeline(self):
        logging.info("---GetSleepTime /statuses/user_timeline ---")

        seconds_to_wait_object = self.api.CheckRateLimit('/statuses/user_timeline')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        logging.info("Seconds to wait for GetSleepTime('/statuses/user_timeline') is: " + str(seconds_to_wait))
        return seconds_to_wait


    # def get_sleep_time_for_get_follower_ids_request(self):
    #     print("---GetSleepTime /followers/ids ---")
    #     logging.info("---GetSleepTime /followers/ids ---")
    #
    #     seconds_to_wait_object = self.api.CheckRateLimit('/statuses/retweeters/ids')
    #     seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
    #     print("Seconds to wait are: " + str(seconds_to_wait))
    #     logging.info("Seconds to wait for GetSleepTime('/followers/ids') are: " + str(seconds_to_wait))
    #     return seconds_to_wait


    def get_sleep_time_for_get_follower_ids_request(self):
        print("---GetSleepTime /followers/ids ---")
        logging.info("---GetSleepTime /followers/ids ---")

        #seconds_to_wait_object = self.api.CheckRateLimit('/followers/ids')
        seconds_to_wait_object = self.api.CheckRateLimit('/users/show/:id')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        print(("Seconds to wait are: " + str(seconds_to_wait)))
        logging.info("Seconds to wait for GetSleepTime('/followers/ids') are: " + str(seconds_to_wait))
        return seconds_to_wait

    # def get_seconds_to_wait(self, seconds_to_wait_object):
    #     if seconds_to_wait_object.remaining > 0:
    #         seconds_to_wait = 0
    #     else:
    #         epoch_timestamp = seconds_to_wait_object.reset
    #         current_timestamp = time.time()
    #         seconds_to_wait = int(epoch_timestamp - current_timestamp + 5)
    #     return seconds_to_wait

    def get_seconds_to_wait(self, seconds_to_wait_object):
        seconds_to_wait = 0
        if seconds_to_wait_object.remaining == 15 and seconds_to_wait_object.limit == 15:
            seconds_to_wait = 15 * 60
        return seconds_to_wait

    def get_sleep_time_for_get_friend_ids_request(self):
        print("---GetSleepTime /friends/ids ---")
        logging.info("---GetSleepTime /friends/ids ---")

        seconds_to_wait_object = self.api.CheckRateLimit('/users/lookup')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        print(("Seconds to wait are: " + str(seconds_to_wait)))
        logging.info("Seconds to wait for GetSleepTime('/friends/ids') are: " + str(seconds_to_wait))
        return seconds_to_wait

    def get_sleep_time_for_get_retweeter_ids_request(self):
        seconds_to_wait_object = self.api.CheckRateLimit('/statuses/retweeters/ids')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        # print("Seconds to wait are: " + str(seconds_to_wait))
        return seconds_to_wait

    def get_sleep_time_for_twitter_status_id(self):
        seconds_to_wait_object = self.api.CheckRateLimit('/statuses/show/:id')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        print(("Seconds to wait are: " + str(seconds_to_wait)))
        return seconds_to_wait

    def get_sleep_time_for_twitter_timeline_request(self):
        seconds_to_wait_object = self.api.CheckRateLimit('/statuses/user_timeline')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        print(("Seconds to wait are: " + str(seconds_to_wait)))
        return seconds_to_wait

    def get_sleep_time_for_get_retweets_request(self):
        seconds_to_wait_object = self.api.CheckRateLimit('/statuses/retweets')
        seconds_to_wait = self.get_seconds_to_wait(seconds_to_wait_object)
        print(("Seconds to wait are: " + str(seconds_to_wait)))
        return seconds_to_wait

    def init_num_of_get_follower_ids_requests(self):
        self.api.num_of_get_follower_ids_requests = 0
        print(("Number of GetFollowerIds requests is: " + str(self.api.num_of_get_follower_ids_requests)))

    def init_num_of_get_friend_ids_requests(self):
        self.api._num_of_get_friend_ids_requests= 0
        print(("Number of GetFriendIds requests is: " + str(self.api._num_of_get_friend_ids_requests)))

    def init_num_of_get_retweeter_ids_requests(self):
        self.api.num_of_get_retweeter_ids_requests = 0
        print(("Number of GetFollowerIds requests is: " + str(self.api.num_of_get_retweeter_ids_requests)))


    def get_num_of_rate_limit_status_requests(self):
        return self.api.get_num_of_rate_limit_status()

    def get_num_of_get_users_requests(self):
        return self.api.get_num_of_get_users_requests()

    def init_num_of_get_users_requests(self):
        self.api.init_num_of_get_users_requests()

    def get_retweets_by_tweet_id(self, tweet_id):
        return self.api.GetRetweets(tweet_id)