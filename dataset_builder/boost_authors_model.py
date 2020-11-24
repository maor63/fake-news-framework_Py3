'''
@author: User
'''
from DB.schema_definition import *
from .author_boost_score import AuthorStatisticsCalculator
from configuration.config_class import getConfig
from preprocessing_tools.abstract_controller import AbstractController


class BoostAuthorsModel(AbstractController):
    """Calculates Boosting score of an author."""

    def __init__(self, db):

        AbstractController.__init__(self, db)

        configInst = getConfig()

    def execute(self, window_start):

        AbstractController.execute(self, window_start)
        self._db.deleteBoostAuth(self._window_start)

        curr_posts = self._db.getPostsListWithoutEmptyRowsByDate(self._window_start, self._window_end)

        stats_calc = AuthorStatisticsCalculator(self._db)
        authors_statistics, post_to_pointers_scores = stats_calc.run_authors_stats_calculation(curr_posts)

        authors_boost_stats_list = []
        total = len(authors_statistics)
        current = 0
        for author, stats in authors_statistics.items():
            current += 1

            print("boost authors model... processing author_boost_stats: " + str(current) + " out of " + str(total))
            if (author):
                found_author = self._db.getAuthorByName(author)
                # We change this line for authors which have the same guid, but different names so some of them not found by the given author name.
                # This is a known bug in VICO.
                if len(found_author) > 0:
                    authorDomain = found_author[0].domain
                    author_guid = found_author[0].author_guid
                    author_boost_stats = Author_boost_stats(window_start=self._window_start,
                                                            window_end=self._window_end, author_name=author,
                                                            author_domain=authorDomain,
                                                            boosting_timeslots_participation_count=stats.boosting_timeslots_participation_count,
                                                            count_of_authors_sharing_boosted_posts=stats.count_of_authors_sharing_boosted_keyposts,
                                                            num_of_pointers=stats.num_of_pointers,
                                                            num_of_pointed_posts=stats.num_of_pointed_keyposts,
                                                            pointers_scores=stats.pointers_scores,
                                                            scores_sum=stats.scores_sum, scores_avg=stats.scores_avg,
                                                            scores_std=stats.scores_std, author_guid=author_guid)
                    authors_boost_stats_list.append(author_boost_stats)

        authors_boost_stats_list = list(set(authors_boost_stats_list))
        self._db.addAuthors_boost_stats(authors_boost_stats_list)

        post_to_pointers_scores_list = []
        duplicatesList = []

        total_post_to_pointers = len(post_to_pointers_scores)
        current_post_to_pointers = 0
        for post, pointers_scores in post_to_pointers_scores.items():
            current_post_to_pointers += 1
            print("boost authors model... processing post to pointers : " + str(
                current_post_to_pointers) + " out of " + str(total_post_to_pointers))
            total_pointer_scores = len(pointers_scores)
            current_pointer_scores = 0

            for pointer, score in pointers_scores.items():
                current_pointer_scores += 1
                print("boost authors model... processing pointer scores: " + str(
                    current_pointer_scores) + " out of " + str(total_pointer_scores))

                temp = []
                post_data = self._db.getPostUsingURL(post, self._window_start, self._window_end)
                if (len(post_data) > 0):
                    post_id = post_data[0].post_id
                    found_author = self._db.getAuthorByName(pointer.author)
                    # We change this line for authors which have the same guid, but different names so some of them not found by the given author name.
                    # This is a known bug in VICO.
                    if len(found_author) > 0:
                        authorDomain = found_author[0].domain
                        temp.extend([post_id, self._window_start, self._window_end, post, authorDomain, pointer.author,
                                     str_to_date(pointer.date)])
                        if temp not in duplicatesList:
                            duplicatesList.append(temp)
                            post_to_pointers = Post_to_pointers_scores(author_name=pointer.author,
                                                                       author_domain=authorDomain, post_id_to=post_id,
                                                                       window_start=self._window_start,
                                                                       window_end=self._window_end, url_to=post,
                                                                       datetime=str_to_date(pointer.date),
                                                                       pointer_score=score)

                            post_to_pointers_scores_list.append(post_to_pointers)

        self._db.addPosts_to_pointers_scores(post_to_pointers_scores_list)

    def cleanUp(self, window_start):
        print('entered clean up..')
        keep_until = window_start - self.keep_results_for
        self._db.session.query(Author_boost_stats).filter(Author_boost_stats.window_start <= keep_until).delete()
        self._db.session.query(Post_to_pointers_scores).filter(
            Post_to_pointers_scores.window_start <= keep_until).delete()
        self._db.session.commit()
        pass


import unittest


class TestBA(unittest.TestCase):
    def setUp(self):
        from preprocessing_tools.tsv_importer import TsvImporter
        from preprocessing_tools.create_authors_table import CreateAuthorTables
        # sys.argv = [sys.argv[0], 'config_test_offline.ini']
        self._db = DB()
        self._db.setUp()
        self._BA = BoostAuthorsModel(self._db)

        self.fdlObject = TsvImporter(self._db)
        self.fdlObject.setUp()

        self._auth = CreateAuthorTables(self._db)

        self._auth.execute(date('2015-04-26 00:00:00'))

    def testDoubleExecute(self):
        self._BA.execute(date('2015-04-26 00:00:00'))
        tcount1 = self._db.session.execute("select count(*) from authors_boost_stats").scalar()

        self._BA.execute(date('2015-04-26 00:00:00'))
        tcount2 = self._db.session.execute("select count(*) from authors_boost_stats").scalar()
        self.assertEqual(tcount1, tcount2)

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()


if __name__ == "__main__":
    print('')
