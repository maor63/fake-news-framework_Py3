
import os

from commons.commons import clean_tweet
from commons.method_executor import Method_Executor


class ClaimPreprocessor(Method_Executor):
    def __init__(self, db):
        super(ClaimPreprocessor, self).__init__(db)
        self._stopwords = []
        stop_word_file = self._config_parser.eval(self.__class__.__name__, "stopwords")
        self.load_stop_words(stop_word_file)

    def clean_posts_content(self):
        posts = self._db.get_posts()
        print("Init post_id to posts dict")
        posts_dict = {p.post_id: p for p in posts}
        # for claim in claims:
        #     posts_dict[claim.guid] = claim
        claim_id_to_comments = self._db.get_claim_tweet_connections()
        preprocessed_posts = []
        i = 0
        for claim_id, post_id in claim_id_to_comments:
            i += 1
            if i % 100 == 0:
                msg = "\rClaim {0}/{1} processed".format(str(i), len(claim_id_to_comments))
                print(msg, end="")
            claim = posts_dict[claim_id]
            comment = posts_dict[post_id]

            title = self._clean_text(claim.content)
            title = self.clean_extra_spaces(title)

            text = self._clean_text(comment.content)
            text = self._clean_title_from_text(text, title)
            comment.content = self.remove_stopwords(self.clean_extra_spaces(text))
            preprocessed_posts.append(comment)

        print()
        self._db.addPosts(preprocessed_posts)

    def clean_extra_spaces(self, title):
        return ' '.join(title.split())

    def _clean_title_from_text(self, text, title):
        # for x in title.split(' '):
        #     if len(x) > 2:
        #         text = text.replace(x, '')
        return text.replace(title, '')

    def _clean_text(self, text):
        text = text.lower()
        text = clean_tweet(text)
        text = str(text).encode('utf-8')
        text = text.replace('"', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ')
        text = text.replace('%', ' ').replace('?', ' ').replace('!', ' ').replace('*', ' ')
        text = text.replace("'", ' ').replace("$", ' ').replace("@", ' ').replace("#", ' ').replace("&", ' ')
        return text

    def remove_stopwords(self, content):
        words = content.split()
        words = [word for word in words if word not in self._stopwords]
        return " ".join(words)

    def load_stop_words(self, stop_word_file):
        # file_path = self._config_parser.get(self.__class__.__name__, "stopwords_file")
        spaces = "\r\n\t "
        if os.path.exists(stop_word_file):
            with open(stop_word_file) as file:
                self._stopwords = set((line.strip(spaces) for line in file.readlines()))
