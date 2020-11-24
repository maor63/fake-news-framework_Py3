
from itertools import islice
from commons.commons import *
from DB.schema_definition import Post, Author, AuthorConnection
from preprocessing_tools.abstract_controller import AbstractController


class FlickrDiffusionGraphLoader(AbstractController):
    def __init__(self, db):
        super(FlickrDiffusionGraphLoader, self).__init__(db)

    def execute(self, window_start=None):
        data_path = 'data/input/data_flicker/data.txt'
        posts, authors, connections = self._read_flickr_data_file(data_path)
        self._db.add_posts_fast(posts)
        self._db.add_authors_fast(authors)
        self._db.add_author_connections_fast(connections)

    def _read_flickr_data_file(self, data_path):
        N = 3
        posts = []
        authors_connections = []
        authors = []
        i = 1
        with open(data_path, 'rt') as rfile:
            image_data_lines = list(islice(rfile, N))
            while image_data_lines:
                print('\rextract photo data {}'.format(i), end='')
                i += 1
                imageId = int(image_data_lines[0])
                post = self.create_post(imageId, image_data_lines)
                posts.append(post)

                connections = image_data_lines[2].rstrip(';\n').split(';')
                fromUserId = -1  # An invalid ID
                for connection in connections:
                    uids = [int(x) for x in connection.split(',')]
                    if len(uids) == 1:  # Following the previous fromUser
                        assert fromUserId != -1, 'Invalid ID'
                        toUserId = uids[0]
                        author = self.create_author(imageId, toUserId)
                        authors.append(author)
                    elif len(uids) == 2:  # Starting a new fromUser
                        fromUserId = uids[0]
                        toUserId = uids[1]
                        from_author = self.create_author(imageId, fromUserId)
                        to_author = self.create_author(imageId, toUserId)

                        authors += [from_author, to_author]

                        con = AuthorConnection()
                        con.source_author_guid = from_author.author_guid
                        con.destination_author_guid = to_author.author_guid
                        con.connection_type = str(imageId)
                        authors_connections.append(con)

                    else:
                        assert False, 'Error line: '

                image_data_lines = list(islice(rfile, N))
        print()
        print('posts: ' + str(len(posts)))
        print('authors: ' + str(len(authors)))
        print('connections: ' + str(len(authors_connections)))
        return posts, authors, authors_connections

    def create_post(self, imageId, image_data_lines):
        post = Post(post_id=str(imageId), domain='flickr', post_osn_id=imageId)
        tags = [x for x in image_data_lines[1].rstrip(';\n').split(';')]
        assert len(tags) == 100, 'The ground-truth annotations should be 100-dim'
        post.post_type = ','.join(tags)
        return post

    def create_author(self, image_id, user_Id):
        return Author(name=str(user_Id), domain=str(image_id), author_guid=str(user_Id))
