from DB.schema_definition import DB
import networkx as nx
from networkx.algorithms import bipartite
import csv
import re
import random
import copy
import time
import datetime


def experiment1(db, extra_round):
    output_file_name = 'timeline_analysis_predictions_round1.csv'
    cursor = db.get_authors_timelines_temp()

    print('retrieving authors and posts')
    author_post_tuples, author_type_dict = create_author_post_and_author_type_tuples(db, cursor)

    print('creating bi-partite graph')
    bi_graph = create_bi_graph(author_post_tuples)
    authors, posts = list(zip(*author_post_tuples))
    del author_post_tuples
    authors = list(set(authors))

    print('creating authors graph')
    authors_projection_graph = bipartite.projected_graph(bi_graph, authors)
    del bi_graph

    print('counting shared posts')
    unlabeled_author_shared_posts_dict = count_shared_posts(author_type_dict, authors_projection_graph)

    #Add an extra iteration to the experiment
    if extra_round:
        output_file_name = 'timeline_analysis_predictions_round2.csv'
        for author, num_shared_posts_with_bad in unlabeled_author_shared_posts_dict.items():
            if unlabeled_author_shared_posts_dict[author] is not None:
                author_type_dict[author] = 'bad_actor'
        unlabeled_author_shared_posts_dict = count_shared_posts(author_type_dict, authors_projection_graph)

    print('writing results to file')
    with open(output_file_name, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in list(unlabeled_author_shared_posts_dict.items()):
            writer.writerow([key, value])


def count_shared_posts(author_type_dict, authors_projection_graph):
    edge_list = authors_projection_graph.edges()
    unlabeled_author_shared_posts_dict = {}

    for edges in edge_list:
        source_author_name = edges[0]
        destination_author_name = edges[1]
        if author_type_dict[source_author_name] == 'None' and author_type_dict[destination_author_name] == 'bad_actor':
            if source_author_name not in unlabeled_author_shared_posts_dict:
                unlabeled_author_shared_posts_dict[source_author_name] = 1
            else:
                unlabeled_author_shared_posts_dict[source_author_name] += 1
        elif author_type_dict[destination_author_name] == 'None' and author_type_dict[source_author_name] == 'bad_actor':
            if destination_author_name not in unlabeled_author_shared_posts_dict:
                unlabeled_author_shared_posts_dict[destination_author_name] = 1
            else:
                unlabeled_author_shared_posts_dict[destination_author_name] += 1
    return unlabeled_author_shared_posts_dict


def experiment4(db):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')
    with open('timeline_experiment2'+st+'.csv', 'w') as output_file:
            output_file.write('Iteration, Total Good, Total Bad, Erased Good, Erased Bad, Number of labeled instances, TP, TN, FP, FN, Correctly Classified, Incorrectly Classified, Accuracy \n')

            num_iterations = 10
            unlabeled_size = 300
            min_post_threshold = 1

            # 1 get manually labeled good and bad actors
            cursor = db.get_cursor_manually_labeled_authors_with_posts()
            authors_posts, authors_types = create_author_post_and_author_type_tuples(db, cursor)

            # 2.1 create a bipartite graph with (author -> post) edges
            bi_graph = create_bi_graph(authors_posts)

            authors, posts = list(zip(*authors_posts))

            #remove duplicate entries
            authors = list(set(authors))

            author_classifications = {}

            overall_tp = 0
            overall_fp = 0
            overall_tn = 0
            overall_fn = 0

            # 2.2 create author to author projection graph
            authors_projection_graph = bipartite.projected_graph(bi_graph, authors)

            edge_list = authors_projection_graph.edges()

            for i in range(num_iterations):

                curr_authors_types = copy.deepcopy(authors_types)
                total_good = 0
                total_bad = 0
                total_good_hidden = 0
                total_bad_hidden = 0
                for author_guid, type in curr_authors_types.items():
                    if type == 'good_actor':
                        total_good += 1
                    else:
                        total_bad += 1

                # 'unlabel' a subset of authors for the current iteration
                for k in range(unlabeled_size):
                    curr_key = random.choice(authors)
                    if curr_authors_types[curr_key]=='good_actor':
                        total_good_hidden += 1
                    else:
                        total_bad_hidden += 1
                    curr_authors_types[curr_key] = 'unlabeled'

                # 2.3 For each edge check timeline overlap between unlabeled and bad actor
                for edges in edge_list:
                    if curr_authors_types[edges[0]] == 'bad_actor' and curr_authors_types[edges[1]] == 'unlabeled':
                        if (edges[1], i) not in list(author_classifications.keys()):
                            author_classifications[(edges[1], i)] = 1
                        else:
                            author_classifications[(edges[1], i)] += 1
                    elif curr_authors_types[edges[0]] == 'unlabeled' and curr_authors_types[edges[1]] == 'bad_actor':
                        if (edges[0], i) not in list(author_classifications.keys()):
                            author_classifications[(edges[0], i)] = 1
                        else:
                            author_classifications[(edges[0], i)] += 1

                # At the end of the current iteration, we compare our results with the actual label of the author.
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                for author_guid, type in curr_authors_types.items():
                    if type == 'unlabeled':
                        if (author_guid,i) in list(author_classifications.keys()):
                            num_posts_shared_with_bad_actors = author_classifications[(author_guid,i)]
                            if num_posts_shared_with_bad_actors >= min_post_threshold: #we say its a bad actor
                                if authors_types[author_guid] == 'bad_actor':
                                    tp += 1
                                else:
                                    fp += 1
                            else: #we say it's a good actor
                                if authors_types[author_guid] == 'bad_actor':
                                    fn += 1
                                else:
                                    tn += 1
                        else: #the author doesn't appear in the classification, this means it didn't share any post with a bad actor
                            if authors_types[author_guid] == 'bad_actor':
                                fn += 1
                            else:
                                tn += 1
                print("\n")
                print("Total good actors: "+str(total_good)+" total bad actors: "+str(total_bad))
                print("Total good actors with erased labels: " + str(total_good_hidden) + " total bad actors with erased labels: " + str(total_bad_hidden))
                print("Iteration: "+str(i)+" - Total number of labeled instances "+str(unlabeled_size))
                print("True Positive: "+str(tp)+ " (Bad actor identified as bad)")
                print("True Negative: " + str(tn) + " (Good actor identified as good)")
                print("False Positive: " + str(fp) + " (Good actor identified as bad)")
                print("False Negative: " + str(fn) + " (Bad actor identified as good)")
                print("Correctly classified instances (TP+TN): "+str(tp+tn))
                print("Incorrectly classified instances (FP+FN): " + str(fp + fn))
                iter_accuracy = (float)(tp+tn)/(tp+tn+fp+fn)
                output_file.write(str(i)+","+str(total_good)+","+str(total_bad)+","+str(total_good_hidden)+","+
                                  str(total_bad_hidden)+","+str(unlabeled_size)+","+str(tp)+","+str(tn)+","+
                                  str(fp)+","+str(fn)+","+str(tp+tn)+","+str(fp + fn)+","+str(iter_accuracy)+" \n")

                overall_tp += tp
                overall_tn += tn
                overall_fp += fp
                overall_fn += fn

            print("----END----")
            print("Overall: \n")
            print("True Positive: " + str(overall_tp))
            print("True Negative: " + str(overall_tn))
            print("False Positive: " + str(overall_fp))
            print("False Negative: " + str(overall_fn))
            print("Correctly classified instances (TP+TN): " + str(overall_tp + overall_tn))
            print("Incorrectly classified instances (FP+FN): " + str(overall_fp + overall_fn))

            accuracy = (float)(overall_tp+overall_tn)/(overall_tp+overall_tn+overall_fp+overall_fn)
            output_file.write("Overall, , , , ," + str(unlabeled_size*num_iterations) + "," + str(overall_tp) + ","
                       + str(overall_tn) + "," + str(overall_fp) + "," + str(overall_fn) + "," +
                       str(overall_tp + overall_tn) + "," + str(overall_fp + overall_fn)+","+str(accuracy)+" \n")

            output_file.write("\n")
            output_file.write("True Positive: Bad actor identified as bad \n")
            output_file.write("True Negative: Good actor identified as good \n")
            output_file.write("False Positive: Good actor identified as bad \n")
            output_file.write("False Negative: Bad actor identified as good \n")
            output_file.write("Correctly classified instances: TP+TN \n")
            output_file.write("Incorrectly classified instances: FP+FN  \n")
            output_file.write("Accuracy: (TP+TN)/(TP+TN+FP+FN)  \n")

def experiment2(db):
    output_file_name = 'timeline_analysis.experiment2.csv'
    output_file_name2 = 'sharedposts.experiment2.csv'

    print('retrieving bad actors timelines')
    cursor = db.get_labeled_bad_actors_timelines_temp()

    author_post_tuples, author_type_dict = create_author_post_and_author_type_tuples(db, cursor)
    authors, posts = list(zip(*author_post_tuples))
    print('creating bi-partite graph')
    bi_graph = create_bi_graph(author_post_tuples)
    del author_post_tuples

    authors = list(set(authors))
    print('creating projection into authors graph')
    authors_projection_graph = bipartite.projected_graph(bi_graph, authors)

    edges_list = authors_projection_graph.edges()
    del authors_projection_graph
    del authors


    manually_labeled_authors_shared_posts_dict = {}

    bought_sub_types = ['crowdturfer','acquired']
    manually_labeled_authors_sub_types = ['spammer','bot','news_feed','company','private']
    interesting_edges = []

    #count how many bought actors we find sharing posts with manually labeled ones/
    print('iterating over author-author edges')
    for edge in edges_list:
        source_author_name = edge[0]
        destination_author_name = edge[1]
        if author_type_dict[source_author_name] in manually_labeled_authors_sub_types and author_type_dict[destination_author_name] in bought_sub_types:
            interesting_edges.append(edge)
            if source_author_name not in manually_labeled_authors_shared_posts_dict:
                manually_labeled_authors_shared_posts_dict[source_author_name] = 1
            else:
                manually_labeled_authors_shared_posts_dict[source_author_name] += 1

        elif author_type_dict[destination_author_name] in manually_labeled_authors_sub_types and author_type_dict[source_author_name] in bought_sub_types:
            interesting_edges.append(edge)
            if destination_author_name not in manually_labeled_authors_shared_posts_dict:
                manually_labeled_authors_shared_posts_dict[destination_author_name] = 1
            else:
                manually_labeled_authors_shared_posts_dict[destination_author_name] += 1

    print('writing results into file')
    with open(output_file_name, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in list(manually_labeled_authors_shared_posts_dict.items()):
            writer.writerow([key, value, author_type_dict[key]])

    with open(output_file_name2, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for item in interesting_edges:
            writer.writerow([item])




def clean_content(post_content):
    content = re.sub(r'https?:.*', '', post_content, flags=re.MULTILINE)
    if len(content) > 10:
        content = str(content.encode('utf-8'))
        return content
    else:
        return None


def create_author_post_and_author_type_tuples(db, cursor):
    tuple_generator = db.result_iter(cursor)
    author_post_tuples = []
    author_type_dict = {}

    for t in tuple_generator:
        content = clean_content(t[1])
        if content is not None:
            author_post_tuples.append([t[0], content])
            author_type_dict[t[0]] = t[2]

    return author_post_tuples, author_type_dict


def create_bi_graph(post_author_tuples):
    bi_graph = nx.Graph()
    bi_graph.add_edges_from(post_author_tuples)

    return bi_graph

if __name__ == '__main__':


    db = DB()
    db.setUp()
    #experiment1(db, extra_round=False)
    #experiment1(db, extra_round=True)
    experiment2(db)




    print("Finished !!!!")




