# encoding: utf-8
# need to be added to the system



import logging

from DB.schema_definition import Post, Author, Politifact_Liar_Dataset, date, Claim
from commons.commons import *
from preprocessing_tools.post_importer import PostImporter
import csv
import os
import pandas as pd


class Politifact_Liar_Dataset_Importer(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        self._data_folder = self._config_parser.eval(self.__class__.__name__, "data_folder")

    def execute(self, window_start=None):
        self._read_liar_dataset()

    def _read_liar_dataset(self):
        self._liar_dataset_records = []
        self._posts = []
        self._authors = []

        file_names = os.listdir(self._data_folder)
        tsv_file_names = [file_name for file_name in file_names if "tsv" in file_name]
        #try:
        for tsv_file_name in tsv_file_names:
            with open(self._data_folder + tsv_file_name, 'rb') as tsvin:
                print("Imported file name is: {0}".format(tsv_file_name))
                rows = csv.reader(tsvin, delimiter='\t')
                self._records_counter = 0
                for row in rows:
                    if len(row) == 14:
                        self._add_row(tsv_file_name, row)

        self._db.addPosts(self._liar_dataset_records)
        self._db.addPosts(self._posts)
        self._db.addAuthors(self._authors)
        #
        # self._records_dataframe = self._records_dataframe.pivot(columns='attribute_name', values='attribute_value', index='original_dataset_id')
        # self._records_dataframe = self._records_dataframe[columns]
        #
        # engine = self._db.engine
        # #engine.text_factory = str
        # self._records_dataframe.to_sql(name="politifact_liar_dataset", con=engine)

        # except IOError:
        #     print('An error occured trying to read the file.')
        #
        # except ValueError:
        #     print('Non-numeric data found in the file.')
        #
        # except ImportError:
        #     print("NO module found")
        #
        # except EOFError:
        #     print('Why did you do an EOF on me?')
        #
        # except KeyboardInterrupt:
        #     print('You cancelled the operation.')
        #
        # except:
        #     print('Problem in file name: {0} and id={1}'.format(file_name, id))

    def _add_row(self, tsv_file_name, row):

        self._records_counter += 1
        original_liar_dataset_id = row[0].replace(".json", "")

        print("\r Import record {0}:{1}".format(self._records_counter, original_liar_dataset_id), end="")
        try:
            # row = row.strip()
            row = list([x.decode('utf-8').strip() for x in row])
            targeted_label = str(row[1])
            statement = str(row[2])
            subject = str(row[3])
            speaker = str(row[4])
            speaker_job_title = str(row[5])
            state_info = str(row[6])
            party_affiliation = str(row[7])
            barely_true_count = str(row[8])
            false_count = str(row[9])
            half_true_count = str(row[10])
            mostly_true_count = str(row[11])
            pants_on_fire_count = str(row[12])
            context = str(row[13])
            dataset_affiliation = str(tsv_file_name.replace(".tsv", ""))
        except Exception as e:
            logging.info("fell while parsing, rip")
            raise Exception("fell while parsing line, rip")

        claim = self._create_claim(original_liar_dataset_id, speaker, targeted_label, statement, subject)
        self._posts.append(claim)

        # author = self._create_author(post, dataset_affiliation)
        # self._authors.append(author)

        post_guid = claim.claim_id

        politifact_liar_dataset_record = self._create_politifact_liar_dataset_record(original_liar_dataset_id, statement, targeted_label,
                                                                                     subject, speaker, speaker_job_title,
                                                                                     state_info, party_affiliation, barely_true_count,
                                                                                     false_count, half_true_count, mostly_true_count,
                                                                                     pants_on_fire_count, context, dataset_affiliation, post_guid)
        self._liar_dataset_records.append(politifact_liar_dataset_record)




    def _create_post(self, original_liar_dataset_id, speaker, targeted_label, statement):
        post = Post()

        post.post_id = str(original_liar_dataset_id)

        post_guid = compute_post_guid(self._social_network_url, original_liar_dataset_id, '2007-01-01 00:00:00')
        post.guid = post_guid
        post.domain = self._domain
        post.author = speaker
        author_guid = compute_author_guid_by_author_name(speaker)
        post.author_guid = author_guid
        post.post_osn_guid = post_guid
        post.date = date('2007-01-01 00:00:00')
        post.post_type = targeted_label

        post.content = statement

        return post

    def _create_claim(self, original_liar_dataset_id, speaker, targeted_label, statement, subject):
        claim = Claim()
        osn_id = str(original_liar_dataset_id)
        claim.claim_id = compute_post_guid(self._social_network_url, osn_id, '2007-01-01 00:00:00')
        claim.domain = 'LiarLiar'
        claim.description = statement
        claim.title = statement
        claim.url = ''
        claim.category = 'Politic'
        claim.sub_category = subject
        claim.verdict = targeted_label
        claim.verdict_date = str_to_date('2007-01-01 00:00:00')
        claim.keywords = ''
        return claim


    def _create_author(self, post, dataset_affiliation):
        author = Author()

        author_name = post.author
        author.name = author_name
        author.author_screen_name = author_name

        author_guid = compute_author_guid_by_author_name(author_name)
        author.author_osn_id = author_guid
        author.author_guid = author_guid
        author.domain = self._domain

        author.author_type = post.post_type
        author.notifications = dataset_affiliation

        return author

    def _create_politifact_liar_dataset_record(self, original_liar_dataset_id, statement, targeted_label, subject,
                                               speaker, speaker_job_title, state_info, party_affiliation,
                                               barely_true_count, false_count, half_true_count, mostly_true_count,
                                               pants_on_fire_count, context, dataset_affiliation, post_guid):

        liar_dataset_record = Politifact_Liar_Dataset()

        liar_dataset_record.original_id = original_liar_dataset_id
        liar_dataset_record.targeted_label = targeted_label
        liar_dataset_record.statement = statement
        liar_dataset_record.subject = subject
        liar_dataset_record.speaker = speaker
        liar_dataset_record.speaker_job_title = speaker_job_title
        liar_dataset_record.state_info = state_info
        liar_dataset_record.party_affiliation = party_affiliation
        liar_dataset_record.barely_true_count = barely_true_count
        liar_dataset_record.false_count = false_count
        liar_dataset_record.half_true_count = half_true_count
        liar_dataset_record.mostly_true_count = mostly_true_count
        liar_dataset_record.pants_on_fire_count = pants_on_fire_count
        liar_dataset_record.context = context
        liar_dataset_record.dataset_affiliation = dataset_affiliation
        liar_dataset_record.post_guid = post_guid

        return liar_dataset_record











