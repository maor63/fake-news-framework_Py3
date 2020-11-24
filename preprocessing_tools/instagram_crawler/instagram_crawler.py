
import logging.config
import time
from datetime import datetime

import requests

from DB.schema_definition import Post, InstagramPost, InstagramAuthor, Author
from commons.commons import compute_post_guid, date_to_str, compute_author_guid_by_author_name
from commons.method_executor import Method_Executor

import json

from preprocessing_tools.instagram_crawler.instagram_endpoints import *

__author__ = "Joshua Grogin"


class InstagramCrawler(Method_Executor):
    def __init__(self, db):
        super(InstagramCrawler, self).__init__(db)
        self._comment_per_submission_limit = self._config_parser.eval(self.__class__.__name__,
                                                                      "max_object_without_saving")
        self.max_entries_without_saving = self._config_parser.eval(self.__class__.__name__,
                                                                   "max_entries_without_saving")
        global CONNECT_TIMEOUT, MAX_RETRIES, MAX_RETRY_DELAY, RETRY_DELAY, MAX_MEDIA_PER_PAGE, MAX_MEDIA_DOWNLOADS
        CONNECT_TIMEOUT = self._config_parser.eval(self.__class__.__name__, 'connect_timeout')
        MAX_RETRIES = self._config_parser.eval(self.__class__.__name__, 'max_retries')
        RETRY_DELAY = self._config_parser.eval(self.__class__.__name__, 'retry_delay')
        MAX_RETRY_DELAY = self._config_parser.eval(self.__class__.__name__, 'max_retry_delay')
        MAX_MEDIA_PER_PAGE = self._config_parser.eval(self.__class__.__name__, 'max_media_per_page')
        MAX_MEDIA_DOWNLOADS = self._config_parser.eval(self.__class__.__name__,
                                                       "max_media_downloads_per_query")
        self.username = 'theacemanofspades'
        self.password = 'joshaceman1@'
        self.insta_crawler = InstagramCrawlerAgent(self.username, self.password)
        self.authors = []
        self.posts = []
        self.comments = []

    # def execute(self, window_start=None):
    #     posts = self.insta_crawler.get_user_posts(528817151)
    #     followedby = self.insta_crawler.get_user_followed_by(528817151)
    #     following = self.insta_crawler.get_user_following(528817151)
    #     comments = self.insta_crawler.get_post_comments_by_short_code('Bw5JHjhg8DQ')
    #     italy = self.insta_crawler.get_posts_by_hashtag('italy')
    #     user = self.insta_crawler.get_profile_info('nasa')
    #     print("Insta done!")
    #     iposts = [self._json_post_to_db_post_converter(post) for post in posts + italy['posts']]
    #     iposts += [self._json_post_to_db_instagram_post_converter(post) for post in posts + italy['posts']]
    #     self._db.addPosts(iposts)
    #     icomments = [self._json_comment_to_db_comment_converter(c) for c in comments]
    #     icomments += [self._json_comment_to_db_instagram_comment_converter(c) for c in comments]
    #     self._db.addPosts(icomments)
    #     to_add = [self._json_user_to_db_author_converter(u) for u in followedby + following + [user]]
    #     to_add += [self._json_user_to_db_instagram_user_converter(u) for u in followedby + following + [user]]
    #     self._db.addPosts(to_add)

    def save_to_db(self):
        self._db.addPosts(self.authors)
        self._db.addPosts(self.posts)
        self._db.addPosts(self.comments)
        release_list(self.authors)
        release_list(self.posts)
        release_list(self.comments)

    def save_to_db_if_max_passed(self, max):
        if len(self.authors) + len(self.posts) + len(self.comments) >= max:
            self.save_to_db()

    def crawl_users_from_usernames_input(self):
        users = self._config_parser.eval(self.__class__.__name__, 'usernames_input')
        self.crawl_users(users)

    def crawl_users_from_db(self):
        users = [user.name for user in self._db.get_authors_by_domain('Instagram_author')]
        self.crawl_users(users)

    def crawl_users(self, users):
        for i, username in enumerate(users):
            print("\rInstagramCrawler crawling user {0} {1}/{2}".format(username, i, len(users)))
            user = self.insta_crawler.get_profile_info(username)
            self.authors.append(self._json_user_to_db_author_converter(user))
            posts = self.insta_crawler.get_user_posts(user['id'])
            self._helper_add_posts_and_commentd(posts)
            release_list(posts)
            following = self.insta_crawler.get_user_following(user['id'])
            for author in following:
                self.authors.append(self._json_user_to_db_author_converter(author))
                self.authors.append(self._json_user_to_db_instagram_user_converter(author))
                self.save_to_db_if_max_passed(self.max_entries_without_saving)
            release_list(following)
            followedby = self.insta_crawler.get_user_followed_by(user['id'])
            for author in followedby:
                self.authors.append(self._json_user_to_db_author_converter(author))
                self.authors.append(self._json_user_to_db_instagram_user_converter(author))
                self.save_to_db_if_max_passed(self.max_entries_without_saving)
            release_list(followedby)
            # self.posts += [self._json_post_to_db_post_converter(post) for post in posts] self.posts += [
            # self._json_post_to_db_instagram_post_converter(post) for post in posts] self.comments += [
            # self._json_comment_to_db_comment_converter(c) for c in comments] self.comments += [
            # self._json_comment_to_db_instagram_comment_converter(c) for c in comments] self.authors += [
            # self._json_user_to_db_author_converter(u) for u in followedby + following + [user]] self.authors += [
            # self._json_user_to_db_instagram_user_converter(u) for u in followedby + following + [user]]
            self.save_to_db()

    def _helper_add_posts_and_commentd(self, posts):
        for post in posts:
            self.posts.append(self._json_post_to_db_post_converter(post))
            self.posts.append(self._json_post_to_db_instagram_post_converter(post))
            comments = self.insta_crawler.get_post_comments_by_short_code(post['shortcode'])
            for comment in comments:
                self.comments.append(self._json_comment_to_db_comment_converter(comment))
                self.comments.append(self._json_comment_to_db_instagram_comment_converter(comment))
                self.save_to_db_if_max_passed(self.max_entries_without_saving)
            release_list(comments)

    def crawl_hashtags_from_hashtags_input(self):
        tags = self._config_parser.eval(self.__class__.__name__, 'hashtags_input')
        for i, tag in enumerate(tags):
            print("\rInstagramCrawler crawling tag {0} {1}/{2}".format(tag, i, len(tags)))
            posts = self.insta_crawler.get_posts_by_hashtag(tag)
            self._helper_add_posts_and_commentd(posts['posts'])
            # for post in posts[u'posts']:
            #     self.posts.append(self._json_post_to_db_post_converter(post))
            #     self.posts.append(self._json_post_to_db_instagram_post_converter(post))
            #     self.save_to_db_if_max_passed(self.max_entries_without_saving)
        self.save_to_db()

    def _json_post_to_db_post_converter(self, post, domain="Instagram_post"):
        rpost = Post()
        rpost.post_osn_id = str(post['id'])
        rpost.created_at = datetime.fromtimestamp(post['taken_at_timestamp'])
        rpost.author = post['owner']['id']
        rpost.author_guid = compute_author_guid_by_author_name(rpost.author)
        rpost.url = str('https://www.instagram.com/p/{}/'.format(post['shortcode']))
        rpost.content = ', '.join(x['node']['text'] for x in post['edge_media_to_caption']['edges'])
        rpost.guid = compute_post_guid(rpost.url, rpost.post_osn_id, date_to_str(rpost.created_at))
        rpost.domain = domain
        rpost.post_type = domain
        rpost.post_id = rpost.guid
        return rpost

    def _json_post_to_db_instagram_post_converter(self, post, domain="Instagram_post"):
        ipost = InstagramPost()
        ipost.id = post['id']
        ipost.comment_count = post['edge_media_to_comment']['count']
        ipost.comments_disabled = post['comments_disabled']
        ipost.dimensions = dimensions = json.dumps(post['dimensions'])
        ipost.display_url = post['display_url']
        ipost.gating_info = gating_info = post.setdefault('gating_info', None)
        ipost.instagram_typename = json.dumps(post['__typename'])
        ipost.is_video = post['is_video']
        ipost.likes = post['edge_media_preview_like']['count']
        ipost.media_preview = json.dumps(post.setdefault('media_preview', 'null'))
        ipost.shortcode = post['shortcode']
        ipost.thumbnail_resources = json.dumps(post['thumbnail_resources'])
        ipost.hashtag = post.setdefault('hashtag', None)
        return ipost

    def _json_comment_to_db_comment_converter(self, post, domain="Instagram_comment"):
        rpost = Post()
        rpost.post_osn_id = str(post['id'])
        rpost.created_at = datetime.fromtimestamp(post['created_at'])
        rpost.author = post['owner']['id']
        rpost.author_guid = compute_author_guid_by_author_name(rpost.author)
        rpost.url = str('https://www.instagram.com/p/{}/'.format(post['shortcode']))
        rpost.content = post['text']
        rpost.guid = compute_post_guid(rpost.url, rpost.post_osn_id, date_to_str(rpost.created_at))
        rpost.domain = domain
        rpost.post_type = domain
        rpost.post_id = rpost.guid
        return rpost

    def _json_comment_to_db_instagram_comment_converter(self, post, domain="Instagram_comment"):
        ipost = InstagramPost()
        ipost.id = post['id']
        ipost.display_url = post['owner']['profile_pic_url']
        ipost.instagram_typename = domain
        ipost.shortcode = post['shortcode']
        return ipost

    def _json_user_to_db_author_converter(self, user, domain='Instagram_author'):
        author = Author()
        author.name = user['username']
        author.author_screen_name = author.name
        author.author_guid = compute_author_guid_by_author_name(author.name)
        author.domain = domain
        author.author_type = domain
        author.author_osn_id = user['id']
        author.author_full_name = user['full_name']
        author.description = user.setdefault('biography', None)
        author.url = 'https://www.instagram.com/' + author.author_screen_name
        author.profile_image_url = user['profile_pic_url']
        return author

    def _json_user_to_db_instagram_user_converter(self, user, domain='Instagram_author'):
        author = InstagramAuthor()
        author.id = user['id']
        author.description = user['biography']
        author.followers_count = user.setdefault('followers_count', None)
        author.following_count = user.setdefault('following_count', None)
        author.is_business_account = user.setdefault('is_business_account', None)
        author.is_joined_recently = user.setdefault('is_joined_recently', None)
        author.is_private = user.setdefault('is_private', None)
        author.posts_count = user.setdefault('posts_count', None)
        return author


class InstagramCrawlerAgent:
    def __init__(self, username, password):
        self.user_data = {}
        self.password = password
        self.username = username
        self.session = requests.Session()
        self.cookies = self.session.cookies
        self.logged_in = False
        self.login(username, password)
        self.last_json = {}

    def login(self, username, password):
        """Logs in to instagram."""
        self.session.headers.update({'Referer': BASE_URL, 'user-agent': STORIES_UA})
        req = self.session.get(BASE_URL)
        self.session.headers.update({'X-CSRFToken': req.cookies['csrftoken']})
        login_data = {'username': self.username, 'password': self.password}
        login = self.session.post(LOGIN_URL, data=login_data, allow_redirects=True)
        self.session.headers.update({'X-CSRFToken': login.cookies['csrftoken']})
        self.cookies = login.cookies
        login_text = json.loads(login.text)
        # print(login_text)
        if login_text.get('authenticated') and login.status_code == 200:
            self.logged_in = True
            self.session.headers = {'user-agent': CHROME_WIN_UA}
        else:
            raise Exception("Instagram login failed")

    def logout(self):
        """Logs out of instagram."""
        if self.logged_in:
            logout_data = {'csrfmiddlewaretoken': self.cookies['csrftoken']}
            self.session.post(LOGOUT_URL, data=logout_data)
            self.logged_in = False

    def get_profile_info(self, username):
        if self.logged_in is False:
            return
        url = ACCOUNT_JSON_INFO.format(username=username)
        response = self.get_json(url)

        if response is None:
            return
        try:
            user_info = json.loads(response.text)['graphql']['user']
            user_info = {
                'username': username,
                'biography': user_info['biography'],
                'followers_count': user_info['edge_followed_by']['count'],
                'following_count': user_info['edge_follow']['count'],
                'full_name': user_info['full_name'],
                'id': user_info['id'],
                'is_business_account': user_info['is_business_account'],
                'is_joined_recently': user_info['is_joined_recently'],
                'is_private': user_info['is_private'],
                'posts_count': user_info['edge_owner_to_timeline_media']['count'],
                'profile_pic_url': user_info['profile_pic_url']
            }
        except (KeyError, IndexError, StopIteration):
            return
        return user_info

    def _get_general_query(self, type_id, id, media_form, container, URL, max_items=MAX_MEDIA_DOWNLOADS):
        items = []
        end_cursor = 'null'
        variables = '"{type_id}":"{id}","first":{max_media},"after":{end_cursor}'
        url = URL.format(variables='{' + variables.format(type_id=type_id, id=id, max_media=MAX_MEDIA_PER_PAGE,
                                                          end_cursor=end_cursor) + '}')
        response = self.get_json(url)
        if response is not None:
            try:
                total = json.loads(response.text)['data'][media_form][container]['count']
            except:
                total = 'unknown'
            i = 0
            while True and response is not None:
                data = json.loads(response.text)['data'][media_form][container]
                items += [item['node'] for item in data['edges']]
                print(
                    "\rInstagramCrawler finished {0}/{1} of user {2}'s {3}".format((i + 1) * MAX_MEDIA_PER_PAGE, total,
                                                                                   id,
                                                                                   container), end='')
                if not data['page_info']['has_next_page'] or i * MAX_MEDIA_PER_PAGE >= max_items:
                    break
                i += 1
                end_cursor = data['page_info']['end_cursor']
                variables = '"{type_id}":"{id}","first":{max_media},"after":"{end_cursor}"'
                url = URL.format(variables='{' + variables.format(type_id=type_id, id=id, max_media=MAX_MEDIA_PER_PAGE,
                                                                  end_cursor=end_cursor) + '}')
                response = self.get_json(url)

        return items

    def _get_user_query(self, user_id, container, URL, max_items=MAX_MEDIA_DOWNLOADS):
        return self._get_general_query('id', user_id, 'user', container, URL, max_items)

    def _get_short_code_query(self, short_code, container, URL, max_items=MAX_MEDIA_DOWNLOADS):
        return self._get_general_query('shortcode', short_code, 'shortcode_media', container, URL, max_items)

    def get_user_posts(self, user_id, max_items=MAX_MEDIA_DOWNLOADS):
        return self._get_user_query(user_id, 'edge_owner_to_timeline_media', ACCOUNT_MEDIAS, max_items)

    def get_user_followed_by(self, user_id, max_items=MAX_MEDIA_DOWNLOADS):
        return self._get_user_query(user_id, 'edge_followed_by', FOLLOWERS_URL_HASH, max_items)

    def get_user_following(self, user_id, max_items=MAX_MEDIA_DOWNLOADS):
        return self._get_user_query(user_id, 'edge_follow', FOLLOWING_URL_HASH, max_items)

    def get_post_comments_by_short_code(self, short_code, max_items=MAX_MEDIA_DOWNLOADS):
        comments = self._get_short_code_query(short_code, 'edge_media_to_comment', COMMENTS_BEFORE_COMMENT_ID_BY_CODE,
                                              max_items)
        for comment in comments:
            comment['shortcode'] = short_code
        return comments

    def get_posts_by_hashtag(self, tag_name, max_items=MAX_MEDIA_DOWNLOADS):
        res = {
            'posts': self._get_general_query('tag_name', tag_name, 'hashtag', 'edge_hashtag_to_media', HASHTAG_URL,
                                              max_items)}
        for p in res['posts']:
            p['hashtag'] = tag_name
        data = json.loads(self.last_json.text)['data']['hashtag']
        res['id'] = data['id']
        res['name'] = data['name']
        res['profile_pic_url'] = data['profile_pic_url']
        return res

    def get_json(self, *args, **kwargs):
        """Retrieve text from url. Return text as string or None if no data present """
        self.last_json = self.safe_get(*args, **kwargs)
        return self.last_json

    def safe_get(self, *args, **kwargs):
        retry = 0
        retry_delay = RETRY_DELAY
        while True:
            try:
                try:
                    response = self.session.get(timeout=CONNECT_TIMEOUT, cookies=self.cookies, *args, **kwargs)
                    if response.status_code == 404:
                        return
                    response.raise_for_status()
                    content_length = response.headers.get('Content-Length')
                    if content_length is not None and len(response.content) != int(content_length):
                        raise requests.exceptions.RequestException('Partial response')
                    return response
                except KeyboardInterrupt:
                    raise
                except requests.exceptions.RequestException as e:
                    if retry < MAX_RETRIES:
                        time.sleep(retry_delay)
                        retry_delay = min(2 * retry_delay, MAX_RETRY_DELAY)
                        retry = retry + 1
                        continue
                    raise e
            except:
                pass


class StubMockup(object):
    def __new__(cls, **attributes):
        result = object.__new__(cls)
        result.__dict__ = attributes
        return result


def release_list(l):
    try:
        del l[:]
        del l
    except:
        pass
