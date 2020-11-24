
import csv
import logging.config
import time
from collections import defaultdict
from datetime import timedelta, datetime
import praw
import prawcore
from praw.models import MoreComments
from DB.schema_definition import Post, Claim_Tweet_Connection, Author, RedditPostCommentConnection, RedditPost, RedditAuthor
from commons.commons import compute_post_guid, date_to_str, compute_author_guid_by_author_name
from commons.method_executor import Method_Executor

__author__ = "Maor Reuben"


class RedditCrawler(Method_Executor):
    def __init__(self, db):
        super(RedditCrawler, self).__init__(db)
        self._number_of_attempts = self._config_parser.eval(self.__class__.__name__, "number_of_attempts") # TODO: extract to config file
        self._max_object_without_saving = self._config_parser.eval(self.__class__.__name__, "max_object_without_saving")
        self.submission_limit = self._config_parser.eval(self.__class__.__name__, "submission_limit")
        self._comment_per_submission_limit = self._config_parser.eval(self.__class__.__name__,
                                                                      "comment_per_submission_limit")
        self.reddit = praw.Reddit(client_id='lEFw7PrfHU6fCA',
                                  client_secret='u08bDqXnbmoazeL4pR4QYaAR3HE',
                                  password='fT56jBnm',
                                  user_agent='reddite_crawler',
                                  username='LinorLipman')
        self.subreddit = self.reddit.subreddit("all")
        self._month_interval = self._config_parser.eval(self.__class__.__name__, "month_interval")
        self._output_folder_full_path = self._config_parser.eval(self.__class__.__name__, "output_folder_full_path")
        self._comments = []
        self._claim_post_connections = []
        self._post_id_tweets_id_before_dict = defaultdict(set)
        self._post_id_tweets_id_after_dict = defaultdict(set)
        self._reddit_request_count = 0
        print(self.reddit.user.me())
        self._redditors = []
        self._deleted_redditors = []
        self._posts = []
        # TODO: add sepreate list for reddit_post (more readable)
        self._claim_tweet_connections = []
        self._post_comment_connections = []
        self._input_claims_file = self._config_parser.eval(self.__class__.__name__, "input_claims_file")

    def get_comments_by_posts_content(self):
        self._get_comments_by_posts_by_method(self._search_comments_by_claim_content)

    def get_comments_by_posts_key_words(self):
        self._get_comments_by_posts_by_method(self._search_comments_by_claim_key_words)

    def export_reddit_data_to_csv(self):
        input_reddit_csv_path = self._input_claims_file
        logging.info("RedditCrawler fetching data from {0}".format(input_reddit_csv_path))

        with open(input_reddit_csv_path, 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)[1:]
            for i, row in enumerate(lines):
                while True:
                    try:
                        claim_id = row[0]
                        for url in row[3:]:
                            if len(url) > 0:
                                self._extract_posts_comments_and_authors(claim_id, url)
                        print("\rRedditCrawler finished {0}/{1}".format(i, len(lines)), end='')
                    except Exception as e:
                        print("\rRedditCrawler {0}/{1} received exception: {2}".format(i, len(lines), e.message), end='')
                        time.sleep(5)
                        continue
                    else:
                        break

        print()
        self.save_to_db()
        logging.info("RedditCrawler finished storing data from {0}".format(input_reddit_csv_path))

    def _extract_posts_comments_and_authors(self, claim_id, url):
        for attempt in range(self._number_of_attempts):
            try:
                reddit_submission = self.reddit.submission(url=url)
                post = self._convert_reddit_post_to_post(reddit_submission)
                reddit_post = self._create_reddit_post(reddit_submission, post.guid)
                reddit_post.is_submitter = True
                reddit_post.link_in_body = getattr(reddit_submission, 'url', None)
                reddit_submission.comments.replace_more(limit=None)
                comments = self._get_posts_from_comments(reddit_submission.comments.list(), reddit_submission)
                self._posts += comments + [post, reddit_post]
                # TODO: check if adding post to connection tables is necessary
                self._claim_tweet_connections += self._get_claim_tweet_connections(claim_id, comments + [post])
                self._post_comment_connections += self.get_reddit_post_comment_connections(comments + [post], post)
                break

            except prawcore.exceptions.Forbidden as e:
                print('URL {} is Forbidden'.format(url))
                break
            except prawcore.exceptions.ServerError as e:
                print('Server overload code 503, save to DB and sleep 30 sec and try again')
                self.save_to_db()
                time.sleep(5)#30)

    def _create_reddit_post(self, reddit_submission, guid):
        post = RedditPost()
        post.post_id = guid
        post.guid = guid
        post.score = getattr(reddit_submission, 'score', 0)

        if hasattr(reddit_submission, 'upvote_ratio') and reddit_submission.author:
            self.calculate_ups_downs_with_upvote_ratio(post, reddit_submission)
        else:
            post.ups = -1
            post.downs = -1
            post.upvote_ratio = -1

        post.number_of_comments = getattr(reddit_submission, 'num_comments', None)
        post.parent_id = getattr(reddit_submission, 'parent_id', None)
        post.stickied = getattr(reddit_submission, 'stickied', False)
        post.is_submitter = getattr(reddit_submission, 'is_submitter', False)
        post.distinguished = getattr(reddit_submission, 'distinguished', None)
        return post

    def calculate_ups_downs_with_upvote_ratio(self, post, reddit_submission):
        post.upvote_ratio = getattr(reddit_submission, 'upvote_ratio', 1)
        post.ups = int(round((post.upvote_ratio * post.score) / (2 * post.upvote_ratio - 1)) if post.upvote_ratio != 0.5
                       else round(post.score / 2))
        post.downs = post.ups - post.score

    def get_reddit_post_comment_connections(self, comments, post):
        comments_connections = []
        for comment_id in [comment.post_id for comment in comments]:
            post_comment_connection = RedditPostCommentConnection()
            post_comment_connection.post_id = post.post_id
            post_comment_connection.comment_id = comment_id
            comments_connections.append(post_comment_connection)
        return comments_connections

    def save_to_db(self):
        self._db.addPosts(self._posts)
        self._db.add_claim_connections(self._claim_tweet_connections)
        self._db.add_claim_connections(self._post_comment_connections)
        authors = []
        reddit_authors = []
        for i, redditor in enumerate(set(self._redditors)):
            for attempt in range(self._number_of_attempts):
                try:
                    self._retrive_reddit_author(authors, i, reddit_authors, redditor)
                    print("\rretrive reddit author {0}/{1}".format(i, len(self._redditors)), end='')
                except prawcore.exceptions.ServerError as e:
                    print('Server overload code 503, save to DB and sleep 30 sec and try again')
                    self.save_to_db()
                    time.sleep(5)  # 30)
                except Exception as e:
                    print('\r retrive reddit author {0}/{1} exception: {2}'.format(i, len(self._redditors), e.message), end='')
            print()
        for i, redditor in enumerate(set(self._deleted_redditors)):
            author = Author()
            author.name = "deleted"
            author.author_guid = compute_author_guid_by_author_name(redditor)
            author.domain = 'reddit'
            author.author_type = 'deleted'
            authors.append(author)

        self._db.add_authors_fast(authors)
        self._db.add_reddit_authors(reddit_authors)
        self._posts = []
        self._claim_tweet_connections = []
        self._redditors = []
        self._deleted_redditors = []
        self._post_comment_connections = []

    def _retrive_reddit_author(self, authors, i, reddit_authors, redditor):
        # TODO: add user link
        # TODO: add deleted author_guid, and add type-deleted
        author = self._convert_reddit_author_to_author(redditor)
        authors.append(author)
        reddit_authors.append(self._create_reddit_author(redditor, author))

    def _get_comments_by_posts_by_method(self, search_method):
        claims = self._db.get_posts_filtered_by_domain(self._domain)
        i = 0
        count = 0
        for claim in claims:
            i += 1
            msg = "\r Retrieving Reddit comments by claims {0}/{1} ".format(i, len(claims))
            print(msg, end="")
            search_method(claim)
            self._check_point()
            count += 1
        self._db.addPosts(self._comments)
        self._db.add_claim_connections(self._claim_post_connections)

        before_dict = self._post_id_tweets_id_before_dict
        after_dict = self._post_id_tweets_id_after_dict
        output_folder_path = self._output_folder_full_path
        claims_dict = {c.post_id: c for c in claims}
        # self._export_csv_files_for_statistics(output_folder_path, before_dict, after_dict, claims_dict)

    def _check_point(self):
        if len(self._comments) >= self._max_object_without_saving:
            self._db.addPosts(self._comments)
            self._db.add_claim_connections(self._claim_post_connections)
            self._comments = []
            self._claim_post_connections = []

    def _search_comments_by_claim_key_words(self, claim):
        self._post_id_tweets_id_before_dict[claim.guid] = set()
        self._post_id_tweets_id_after_dict[claim.guid] = set()
        for key_word in claim.tags.split(','):
            self._search_query_with_time_interval(key_word.strip(), claim.date, claim.guid)

    def _search_comments_by_claim_content(self, claim):
        self._post_id_tweets_id_before_dict[claim.guid] = set()
        self._post_id_tweets_id_after_dict[claim.guid] = set()
        self._search_query_with_time_interval(claim.content, claim.date, claim.guid)

    def _search_query(self, query):
        assert (isinstance(self.reddit, praw.Reddit))
        submissions = self.subreddit.search(query, limit=self.submission_limit)
        comments = []
        for submission in submissions:
            self.append_comments(submission, 1)
        return comments

    def _search_query_with_time_interval(self, content, publish_date, origin_post_id):
        self._reddit_request_count += 1
        try:
            submissions = self.subreddit.search(content, limit=self.submission_limit)
            count = 0
            comment_count = 0
            for submission in submissions:
                self.append_comments(submission, origin_post_id, publish_date)
                count += 1
                comment_count += len(self._comments)

                if comment_count > self._comment_per_submission_limit:
                    break

        except prawcore.exceptions.ServerError as e:
            self._reddit_request_count = 0
            print()
            print("Error: server overload")
            print("sleep 60 sec")
            time.sleep(60)
            self._search_query_with_time_interval(content, publish_date, origin_post_id)

    def append_comments(self, submission, origin_post_id, publish_date):
        datetime_object = publish_date
        month_interval = timedelta(self._month_interval * 365 / 12)
        start_date = time.mktime((datetime_object - month_interval).timetuple())
        end_date = time.mktime((datetime_object + month_interval).timetuple())
        publish_date = time.mktime(publish_date.timetuple())

        for comment in submission.comments:
            if isinstance(comment, MoreComments):
                continue
            if comment.created > end_date or comment.created < start_date:
                continue

            created_at = datetime.fromtimestamp(comment.created)
            url = str(submission.url + comment.id)
            comment_guid = compute_post_guid(url, comment.id, date_to_str(created_at))

            if start_date < comment.created <= publish_date:
                if comment.score >= 0 and len(comment.body.split(' ')) > 3:
                    self.convert_comment_to_post(comment, submission)
                    self._post_id_tweets_id_before_dict[origin_post_id].add(comment_guid)
            elif publish_date < comment.created <= end_date:
                if comment.score >= 0 and len(comment.body.split(' ')) > 3:
                    self.convert_comment_to_post(comment, submission)
                    self._post_id_tweets_id_after_dict[origin_post_id].add(comment_guid)

    def convert_comment_to_post(self, comment, submission, domain="Reddit"):
        post = Post()
        post.post_osn_id = str(comment.id)
        post.created_at = datetime.fromtimestamp(comment.created)
        post.date = datetime.fromtimestamp(comment.created)
        if hasattr(comment, 'author') and comment.author:
            post.author = str(comment.author.name)
            self._redditors.append(comment.author)
        else:
            self._deleted_redditors.append(str(post.date))
            post.author = str('')
        post.author_guid = compute_author_guid_by_author_name(post.author)
        post.url = str('https://www.reddit.com' + '/'.join(getattr(comment, 'permalink', '').split('/')[3:7]))
        post.title = str(submission.title)
        post.content = str(getattr(comment, 'body', '').encode('utf-8').strip())
        post.guid = compute_post_guid(post.url, post.post_osn_id, date_to_str(post.created_at))
        post.domain = domain
        post.post_type = domain
        post.post_id = post.guid
        post.url = 'https://www.reddit.com{}'.format(comment.permalink)
        return post

    def get_claim_tweet_connection(self, claim_id, post_id):
        claim_tweet_connection = Claim_Tweet_Connection()
        claim_tweet_connection.claim_id = claim_id
        claim_tweet_connection.post_id = post_id
        return claim_tweet_connection

    def _get_posts_from_comments(self, comments, submission):
        posts = []
        for comment in comments:
            if not isinstance(comment, MoreComments):
                post = self.convert_comment_to_post(comment, submission, 'reddit_comment')
                reddit_post = self._create_reddit_post(comment, post.guid)
                posts += [post, reddit_post]
        return posts

    def _get_claim_tweet_connections(self, claim_id, posts):
        return [self.get_claim_tweet_connection(claim_id, post.post_id) for post in posts]

    def _convert_reddit_post_to_post(self, submission):
        post = self.convert_comment_to_post(submission, submission, 'reddit_post')
        # post.url = u'https://www.reddit.com/comments/{}'.format(post.post_osn_id)
        return post

    def _convert_reddit_author_to_author(self, redditor):
        author = Author()
        author.name = getattr(redditor, 'name', '')
        author.author_screen_name = author.name
        author.author_guid = compute_author_guid_by_author_name(author.name)
        author.domain = 'reddit'
        author.created_at = datetime.fromtimestamp(getattr(redditor, 'created_utc', 0))
        author.author_osn_id = getattr(redditor, 'id', '')
        author.author_full_name = getattr(redditor, 'fullname', '')
        author.url = 'https://www.reddit.com/user/' + redditor.name
        return author

    def _create_reddit_author(self, redditor, author):
        reddit_author = RedditAuthor()
        reddit_author.name = author.name
        reddit_author.author_guid = author.author_guid
        try:
            # comments_count = getattr(redditor, 'comments', [])
            reddit_author.comments_count = None  # 0 if comments_count == [] else len(list(comments_count.new(limit=25)))
            reddit_author.comment_karma = getattr(redditor, 'comment_karma', 0)
            reddit_author.link_karma = getattr(redditor, 'link_karma', 0)
            reddit_author.is_gold = getattr(redditor, 'is_gold', False)
            reddit_author.is_moderator = getattr(redditor, 'is_mod', False)
            reddit_author.is_employee = getattr(redditor, 'is_employee', False)
        except Exception as e:
            logging.info("RedditCrawler getting data from user {0}: {1}".format(author.name, e.message))
        return reddit_author
