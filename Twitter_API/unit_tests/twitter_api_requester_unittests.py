# Created by Aviad on 29-Apr-16 12:55 PM.
import unittest

from twitter import TwitterError

from Twitter_API.twitter_api_requester import TwitterApiRequester


class TestTwitterApiRequester(unittest.TestCase):

    def setUp(self):
        app_number = 2
        self._twitter_api_requester = TwitterApiRequester(app_number)

    def testCredentialsAreValid(self):
        authenticated_user = self._twitter_api_requester.verify_credentials()
        self.assertIsNotNone(authenticated_user)
        self.assertIsNotNone(authenticated_user.followers_count)
        self.assertIsNotNone(authenticated_user.friends_count)

    def testCheckRequests(self):
        user = self._twitter_api_requester.get_user_by_screen_name('Jerusalem_Post')
        expected_user_id = 19489239
        self.assertEqual(user.id, expected_user_id)
        user = self._twitter_api_requester.get_user_by_user_id(expected_user_id)
        self.assertEqual(str(user.screen_name), 'Jerusalem_Post')
        # followers = self._twitter_api_requester.get_follower_ids_by_user_id(19489239)
        # self.assertGreater(len(followers), 0)

