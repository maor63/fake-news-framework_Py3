import datetime
import collections
from collections import defaultdict, Counter

#from commons_text import utf8open
import math
import numpy
from commons import * 
from DB.schema_definition import *
from configuration.config_class import getConfig
import logging
import sys
import unicodedata
from functools import reduce


def str_to_dates(datestring):
    return datetime.datetime.strptime(datestring, "%d/%m/%Y %H:%M:%S")

# Timeslot - representing a time window that starts at start_time and ends at end_time. 
Timeslot = collections.namedtuple('Timeslot', 'start_time end_time')
PointingRecord = collections.namedtuple('PointingRecord', 'date guid author')

AuthorStatistics = collections.namedtuple('AuthorStatistics', 'author boosting_timeslots_participation_count count_of_authors_sharing_boosted_keyposts num_of_pointers num_of_pointed_keyposts pointers_scores scores_sum scores_avg scores_std')

class TimeslotUtils:



    """
    Given the start time, create a list of time windows where the last time window ends at last_date.
    last_date is given as a program parameter in the execution command (required format: d/m/Y H:M:S).
    The lengths of the time windows grow according to the formula: a_{n+1} = a_n + n
    """
    def prepare_timeslots(self, start):
        configInst = getConfig() 

        slot_length = int(getConfig().get("BoostAuthorsModel", "seconds_in_slot_unit"))
        seconds_in_slot_unit = int(getConfig().get("BoostAuthorsModel", "boost_jump"))
        window_size = datetime.timedelta(seconds=int(configInst.get("DEFAULT","window_analyze_size_in_sec")))
        
        
        start = unicodedata.normalize('NFKD', start).encode('ascii','ignore')
        last_date = str_to_date(start)+window_size
        
        timeslots = list()

        current_slot_start = str_to_date(start)
        index = 1
        while current_slot_start.date() < last_date.date() + datetime.timedelta(days=1):

            delta = datetime.timedelta(seconds=(slot_length*seconds_in_slot_unit))
            delta = delta * index

            timeslot = Timeslot(current_slot_start, current_slot_start + delta - datetime.timedelta(seconds=1))
            timeslots.append(timeslot)

            current_slot_start = current_slot_start + delta
            index = index + 1

        return timeslots

    """
    Given a list of time windows and a date-time object, return the time window which the given date-time belongs to. 
    """
    def find_matching_timeslot(self, timeslots, date_time):

        for timeslot in timeslots:
            if date_time >= timeslot.start_time and date_time <=timeslot.end_time:
                return timeslot

        return 0

"""
The heuristic represented by this class is as follows:
Assume author A referenced a post p at time window t.
Assume that at time window t+1 has been a "jump" in the number of references to post p.
If the "jump" is higher than a given threshold, author A is to be rewarded.
"""
class HeuristicJumpInRefsCount:



    """
    timeslot_to_referencing_authors hold a dictionary where the key is a time window 
    and the value is a set of authors that referenced a specific post at this time window.
    Note, that each author appears only at the first time window at which she references the specific post.
    [Assume an author referenced a post several times (at different time windows and\or in the same time window). 
    The author will appear only once: at the time window of its earliest reference to the specific post.]
    
    This function collects and returns {time window t --> set(authors)} such that there is a jump in number of references 
    in the succeeding time window: {t | len([t+1]->authors) - len([t]->authors) > boost_jump}.    
    """
    def get_post_boosting_timeslots(self, timeslot_to_referencing_authors):
        configInst = getConfig()
        boost_jump = int(configInst.get("BoostAuthorsModel", "boost_jump"))

        boosting_timeslots_to_authors = defaultdict(set)
        sorted_timeslots = sorted(timeslot_to_referencing_authors.items())

        prev_timeslot, prev_authors = sorted_timeslots[0]
        prev_authors_count = len(prev_authors)

        for (current_timeslot, current_authors) in sorted_timeslots[1:]:
            current_authors_count = len(current_authors)

            if current_authors_count - prev_authors_count > boost_jump:
                boosting_timeslots_to_authors[prev_timeslot] = prev_authors

            #next step
            prev_timeslot = current_timeslot
            prev_authors = current_authors
            prev_authors_count = current_authors_count

        return boosting_timeslots_to_authors

"""
The heuristic represented by this class is as follows:
Assume author A referenced a post p at time window t.
Author A is considered influential if post p has been referenced by many other authors 
at the following time windows.

To compute A's influence on the popularity of a post p, we construct the array 
'timeslots_accumulated_counts'.
This array holds for each time window t the number of distinct authors that referenced p until (and including) time window t.

Assume a post p that was referenced only by:
1) a bot x that referenced the p every timeslot, and
2) a human author A that referenced p at timeslot t=10.
            
Since 'timeslots_accumulated_counts' ignores all x's references except for the first one:
* The score of each of x's references published before t=10, will account only for A's reference 
while ignoring x's references (except for x's first reference). 
** The score of each of x's references published from t=10 onward will be zero, 
since there are no legitimate references to account for at t>10.
*** The score of A's reference (published at t=10) will also be zero.
            
To avoid inflating x's boost impact score by the multiple scores of its references to p,
and to "punish" x for multiple references to p:
account only for the average score among the scores of references to p:
BoostScore(x) = \sum_{p} avg({Score(x's ref to p)} 
"""
class HeuristicInfluenceOverTime:

    timeslotUtils = TimeslotUtils()

    """
    This function constructs a data structure that holds for each time window t
    the number of distinct authors that referenced p until (and including) time window t.
    """
    def get_accumulated_counters(self, post_timeslots_authors, total_ref_count):

        # {time window --> set of authors} 
        sorted_timeslots_to_authors = sorted(post_timeslots_authors.items())

        accumulated_counts = Counter()
        current_accum_count = total_ref_count

        for (ts, authors) in reversed(sorted_timeslots_to_authors):
            accumulated_counts[ts] = current_accum_count
            current_accum_count -= len(authors)

        return accumulated_counts

    """
    This function implements the calculation of Score(pointer->post).
    t - time window
    t.start and t.end - the start time and the end time, respectively, of the time window t.
    tp - the given pointer's time window, i.e. tp.start < pointer.date and pointer.date < tp.end.   
    Score(pointer->post) = \sum_{t.start>=tp.start} (accum[t]-accum[tp])/(t.end-tp.end)^2
    """
    def get_pointer_score(self, pointer, total_ref_count, timeslots_accumulated_counts, timeslots):
        configInst = getConfig() 
        seconds_in_slot_unit = int(configInst.get("BoostAuthorsModel", "seconds_in_slot_unit"))

        sorted_timeslots = sorted(timeslots_accumulated_counts.items())

        p_timeslot = self.timeslotUtils.find_matching_timeslot(timeslots, str_to_date(pointer.date))
        accum_p_timeslot = timeslots_accumulated_counts[p_timeslot]

        score = 0

        for timeslot in reversed(sorted_timeslots):
            addition = timeslot[1] - accum_p_timeslot

            time_delta = abs(timeslot[0].end_time - p_timeslot.end_time).total_seconds() / seconds_in_slot_unit
            time_delta = math.pow(time_delta, 2.0)

            if addition==0 and time_delta==0:
                score += 0
            else:
                score += float(addition)/time_delta

            if p_timeslot.start_time==timeslot[0].start_time and p_timeslot.end_time==timeslot[0].end_time:
                break

        return score

class AuthorStatisticsCalculator:

        
    AMOUNT = None#50
    timeslotUtils = TimeslotUtils()
    jumpHeuristic = HeuristicJumpInRefsCount()
    influenceHeuristic = HeuristicInfluenceOverTime()

    def __init__(self,db):
        self._db = db 


    """
    This function returns a dictionary where the key is a referenced (pointed at) post p and 
    the value is a list of records pointing at (referencing) p.
    """
    def get_reverse_pointers(self, list_of_input_records):
        rev = defaultdict(list)

        for post in list_of_input_records[:self.AMOUNT]:
            guid = post[2]
            date = post[5]
            author = post[1]
            pointing_record = PointingRecord(date, guid, author)
            references = self._db.getReferencesFromPost(post[0])
            for reference in references:
                rev[reference].append(pointing_record)
        if '' in rev:
            del rev['']

        return rev

    """
    This function returns for a post the number of distinct authors that referenced p.
    
    Input:
    post_timeslots_authors - for each post p and each time window t holds the authors that referenced p in time window t.
    """
    def get_total_ref_count(self, post_timeslots_authors):
        all_distinct_authors = reduce(set.union, [authors for (timeslot, authors)
                                                  in post_timeslots_authors.items()])
        return len(all_distinct_authors)

    """
    Return the authors that referenced each post in each timeslot.
    If an author already referenced the post in previous timeslots, it won't be added again.
    
    Input:
    reverse_pointers: reference --> pointers to the reference.
    
    Returns:
    post_timeslot_authors --> { post --> timeslot --> set of authors }
    """
    def get_posts_timeslots_to_authors(self, reverse_pointers):

        post_timeslot_authors = defaultdict(lambda:defaultdict(set))

        for post, refs in reverse_pointers.items():

            if len(refs)>0:
                distinct_authors_till_now = set()

                sorted_pointers = sorted(refs)
                earliest_pointer = sorted_pointers[0]
                timeslots = self.timeslotUtils.prepare_timeslots(earliest_pointer.date)
                for pointer in sorted_pointers:
                   
                    timeslot = self.timeslotUtils.find_matching_timeslot(timeslots, str_to_date(pointer.date))

                    if pointer.author not in distinct_authors_till_now:
                        post_timeslot_authors[post][timeslot].add(pointer.author)
                        distinct_authors_till_now.add(pointer.author)

        return post_timeslot_authors

    def get_jump_in_references_stats(self, posts_refs, post_timeslot_authors):

        # Data structures used for Jump-In-References-Count statistics.
        authors_to_refed_posts = defaultdict(set)

        for post, post_timeslots in post_timeslot_authors.items():

            # for post find boosting timeslots
            # {boosting_timeslot --> set(authors)}
            boosting_timeslots_to_authors = self.jumpHeuristic.get_post_boosting_timeslots(post_timeslots)
            # given boosting timeslots, get all authors that referenced the post in the boosting timeslot
            # count for authors_to_boosting_ts_counter the number of boosted posts  

            for bts, authors in boosting_timeslots_to_authors.items():
                for author in authors:
                    authors_to_refed_posts[author].add(post)

        author_to_authors_referencing_same_posts = defaultdict(set)
        for author, posts in authors_to_refed_posts.items():
            for post in posts:
                author_to_authors_referencing_same_posts[author] |= reduce(set.union, [authors for (timeslot, authors) in post_timeslot_authors[post].items()])

        author_to_authors_referencing_same_posts = {author: len(authors)
                                                    for author, authors in
                                                    author_to_authors_referencing_same_posts.items()}

        return (authors_to_refed_posts, author_to_authors_referencing_same_posts)

    def get_influence_over_time_stats(self, posts_refs, post_timeslot_authors):
        # {author --> pointer --> post --> score(pointer->post)}
        author_pointer_post_boost_impact = defaultdict(lambda:defaultdict(Counter))
        # {author --> post --> pointer --> score(pointer->post)}
        author_post_pointer_boost_impact = defaultdict(lambda:defaultdict(Counter))

        for post, post_timeslots in post_timeslot_authors.items():
            timeslots = self.timeslotUtils.prepare_timeslots(sorted(posts_refs[post])[0].date)

            """
            Assume a post k that was referenced only by:
            1) a bot x that referenced the k every timeslot, and
            2) a human author a that referenced k at timeslot t=10.
            
            Since 'timeslots_accumulated_counts' ignores all x's references except for the first one:
            * The score of each of x's references published before t=10, will account only for a's reference 
            while ignoring x's references (except for x's first reference). 
            ** The score of each of x's references published from t=10 onward will be zero, 
            since there are no legitimate reference to account for at t>10.
            *** The score of a's reference (published at t=10) will also be zero.
            
            To avoid inflating x's boost score by the multiple scores of its references to k
            account only for the maximal score among the scores of references to k:
            BoostScore(x) = \sum_{k} max({Score(x's ref to k)} 
            """
            # Count the distinct authors that referenced the given post.            
            total_ref_count = self.get_total_ref_count(post_timeslots)
            # For each timeslot t count the distinct authors that referenced the given post until t (including t).
            timeslots_accumulated_counts = self.influenceHeuristic.get_accumulated_counters(post_timeslots, total_ref_count)
            for pointer in posts_refs[post]:
                pointer_score = self.influenceHeuristic.get_pointer_score(pointer, total_ref_count, timeslots_accumulated_counts, timeslots)

                author_pointer_post_boost_impact[pointer.author][pointer][post] = pointer_score
                author_post_pointer_boost_impact[pointer.author][post][pointer] = pointer_score

        return (author_pointer_post_boost_impact, author_post_pointer_boost_impact)


    def run_authors_stats_calculation(self, list_of_input_records):

        rev = self.get_reverse_pointers(list_of_input_records)
        post_timeslot_authors = self.get_posts_timeslots_to_authors(rev)

        (authors_to_refed_posts, author_to_authors_referencing_same_posts) = self.get_jump_in_references_stats(rev, post_timeslot_authors)
        (author_pointer_post_boost_impact, author_post_pointer_boost_impact) = self.get_influence_over_time_stats(rev, post_timeslot_authors)

        authors_statistics = defaultdict(AuthorStatistics)

        post_to_pointers_scores = defaultdict(lambda:defaultdict(Counter))

        for author, author_pointers in author_pointer_post_boost_impact.items():
            num_of_pointers = len(author_pointers)

            pointed_posts = author_post_pointer_boost_impact[author]
            num_of_pointed_posts = len(pointed_posts)

            boost_impact_scores = []

            for (post, pointers) in pointed_posts.items():
                boost_impact_scores.append(numpy.mean(list(pointers.values())))
                for pointer in pointers:
                    post_to_pointers_scores[post][pointer] = author_post_pointer_boost_impact[author][post][pointer]

            pointers_scores_string = "|".join(map(str, boost_impact_scores))

            #compute the standard deviation of scores of the author's pointers.
            scores_sum = sum(boost_impact_scores)
            scores_avg = numpy.mean(boost_impact_scores)
            scores_std = numpy.std(boost_impact_scores)

            a_to_as_referencing_same_posts = 0
            if author in author_to_authors_referencing_same_posts:
                a_to_as_referencing_same_posts = author_to_authors_referencing_same_posts[author]

            authorStats = AuthorStatistics(author,
                                           len(authors_to_refed_posts[author]), #authors_to_boosting_ts_counter[author], # Jump heuristic
                                           a_to_as_referencing_same_posts, # Jump heuristic
                                           num_of_pointers,
                                           num_of_pointed_posts,
                                           pointers_scores_string, # Influence over time heuristic
                                           scores_sum, # Influence over time heuristic
                                           scores_avg, # Influence over time heuristic
                                           scores_std) # Influence over time heuristic

            authors_statistics[author] = authorStats

        return authors_statistics, post_to_pointers_scores


#@todo: add test case!!!


# if __name__ == "__main__":
# 
#     configInst = getConfig()
# 
#     pipeline_input = configInst.getValue("Inputs", "input_path")
#     #list_of_input_records = AuthorRecordsReader.read_records(pipeline_input)
#     list_of_input_records = getPostsListWithoutEmptyRows()
#     print "Read %d records from %s" % (len(list_of_input_records), pipeline_input)
# 
#     authors_stats_file = configInst.getValue("Search4Boosters", "authors_boost_stats_file")
# 
#     stats_calc = AuthorStatisticsCalculator()
#     authors_statistics, post_to_pointers_scores = stats_calc.run_authors_stats_calculation(list_of_input_records)
# 
#     with utf8open(authors_stats_file, 'w') as o:
# 
#         header = ["author",
#                   "boosting_timeslots_participation_count",
#                   "count_of_authors_sharing_boosted_posts",
#                   "num_of_pointers",
#                   "num_of_pointed_posts",
#                   "pointers_scores",
#                   "scores_sum",
#                   "scores_avg",
#                   "scores_std"]
# 
#         o.write("\t".join(header) + "\n")
# 
#         for author, stats in authors_statistics.iteritems():
#             o.write("%s\t%d\t%d\t%d\t%d\t%s\t%f\t%f\t%f\n" % (author,
#                                                               stats.boosting_timeslots_participation_count,
#                                                               stats.count_of_authors_sharing_boosted_keyposts,
#                                                               stats.num_of_pointers,
#                                                               stats.num_of_pointed_keyposts,
#                                                               stats.pointers_scores,
#                                                               stats.scores_sum,
#                                                               stats.scores_avg,
#                                                               stats.scores_std)
#             )
#     with utf8open("data/output/post_to_pointers_scores.tsv", 'w') as o:
# 
#         header = ["post",
#                   "author",
#                   "datetime",
#                   "pointer_score"]
# 
#         o.write("\t".join(header) + "\n")
#         for post, pointers_scores in post_to_pointers_scores.iteritems():
#             for pointer, score in pointers_scores.iteritems():
#                 o.write("%s\t%s\t%s\t%f\n" % (post,
#                                               pointer.author,
#                                               pointer.date.strftime("%d/%m/%Y %H:%M:%S"),
#                                               score)
#                 )
            