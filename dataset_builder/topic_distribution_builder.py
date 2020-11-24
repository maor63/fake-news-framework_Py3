from preprocessing_tools.abstract_controller import AbstractController
from commons.commons import *
from DB.schema_definition import SinglePostByAuthor

class TopicDistributionBuilder(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)

    def set_up(self):
        pass

    def execute(self, window_start=None):
        users_posts = self._db.get_posts_by_domain(self._domain)

        new_posts = []
        for user_posts, posts in users_posts.items():
            single_post = ""
            author_guid = user_posts
            post_creation_date = ""
            for post in posts:
                post_creation_date = post.date
                single_post += " "+post.content
            post = self.create_post(single_post, author_guid, post_creation_date)
            new_posts.append(post)
        self._db.addPosts(new_posts)

    def create_post(self, content, author_guid, post_creation_date):
        author_guid = str(author_guid)
        post_id = author_guid
        date = post_creation_date
        content = str(content)
        domain = self._domain

        post = SinglePostByAuthor(post_id=post_id, author_guid=author_guid, date=date, content=content, domain=domain)
        return post