
from webbrowser import open_new_tab
# from scipy.misc import imread
import csv
import logging
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd
from langdetect import detect
import json

class TopicDistrobutionVisualizationGenerator(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._read_predictions_from_db = self._config_parser.eval(self.__class__.__name__, "read_predictions_from_db")
        self.prediction_csv_path = self._config_parser.eval(self.__class__.__name__, "prediction_csv_path")
        self._viz_output_path = self._config_parser.eval(self.__class__.__name__, "output_dir")
        self._buckets = self._config_parser.eval(self.__class__.__name__, "buckets")
        self._target_author_field = self._config_parser.eval(self.__class__.__name__, "target_author_field")
        self._targeted_class_field_name = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_name")
        self._include_unlabeled_predictions = self._config_parser.eval(self.__class__.__name__,
                                                                       "include_unlabeled_predictions")
        self._include_labeled_authors_in_visualization = self._config_parser.eval(self.__class__.__name__,
                                                                                  "include_labeled_authors_in_visualization")
        self._optional_classes = self._config_parser.eval(self.__class__.__name__, "optional_classes")
        self._extend_posts_by_retweets = self._config_parser.eval(self.__class__.__name__, "extend_posts_by_retweets")
        # if you have problem with config - this is it
        # font = "topic_distribution_visualization/Mukadimah.ttf"
        self._font_path = self._config_parser.eval(self.__class__.__name__, "font_path")
        self.include_topic_labels = self._config_parser.eval(self.__class__.__name__, "include_topic_labels")
        self.claim_to_topic_connection_json = self._config_parser.eval(self.__class__.__name__, "claim_to_topic_connection_json")
        self.topic_classification_table_args = self._config_parser.eval(self.__class__.__name__, "topic_classification_table_args")
        self._topic_to_type = {}

    def setUp(self):
        pass

    def execute(self, window_start=None):
        if self.include_topic_labels:
            table_name = self.topic_classification_table_args['table']
            table_id = self.topic_classification_table_args['id']
            type_field = self.topic_classification_table_args['type_field']
            table_id_table_elements_dict = self._db.get_table_dictionary(table_name, table_id)
            claim_to_topic_json = open(self.claim_to_topic_connection_json)
            claim_to_topic_connection = json.load(claim_to_topic_json)
            for claim_id, topic_id in claim_to_topic_connection.items():
                self._topic_to_type[str(topic_id)] = getattr(table_id_table_elements_dict[claim_id], type_field)
            claim_to_topic_json.close()

        self.create_topic_data_list()

    def red_to_green_colors(self, number_of_colors):
        red = 255
        green = 0
        step_size = 255 / number_of_colors
        color_list = []
        while green < 255:
            green += step_size
            red -= step_size
            if green > 255:
                green = 255
            if red < 0:
                red = 0
            color_list.append('#%02x%02x%02x' % (red, green, 0))
        return color_list

    def create_dataset_for_chart(self, rings_data_list_dict):
        html = """datasets: [
            %s
        ]
        """
        data_set = ""
        for ring_data in rings_data_list_dict['datasets']:
            temp = """
                {
                    data: %s,
                    backgroundColor: %s,
                },
                """
            data_set = data_set + temp % (str(ring_data['data']), str(ring_data['backgroundColor']))
        html = html % data_set
        return html

    def create_charts_script(self, topics_and_data):
        i = 1
        script = ""
        for topic in topics_and_data:
            config = """var config%s = {
                            type: 'doughnut',
                            data: {
                                %s
                                ,
                                labels:
                                %s
                            },
                            options: {
                                responsive: true,
                                title: {
                                    display: true,
                                    text: %s
                                },
                                 legend: {
                                    display: false,
                                }
                            }
                        };
                        """
            verdict = ''
            if self.include_topic_labels:
                verdict = self._topic_to_type[topic[0]]
            top = '"Topic :' + topic[0] + ' Verdict: {0}"'.format(verdict)
            script += config % (str(i), self.create_dataset_for_chart(topic[1]), str(topic[2]), top)
            i += 1
        script += "\nwindow.onload = function (){\n"
        for j in range(1, len(topics_and_data) + 1):
            temp = """
                var ctx%s = document.getElementById("chart-area%s");
                var myDoughnutChart = new Chart(ctx%s, config%s);
            """
            script += temp % (str(j), str(j), str(j), str(j))
        script += "}"
        return script

    def create_body_section(self, number_of_topicts):
        script = """<body>
        """
        for i in range(1, number_of_topicts + 1):
            temp = """
                    <div id="canvas-holder%s" style="width:%s">
                        <canvas id="chart-area%s" width="100px" height="100px" />
                    </div>
                    """
            script += temp % (str(i), '20%', str(i))
        script += "</body>"
        return script

    def create_html_file(self, topic_and_data):
        file_name = "topic_visualization"
        filename = self._viz_output_path + file_name + '.html'
        f = open(filename, 'w')
        wrapper = """ <!DOCTYPE HTML>
            <html>
            <head>
                %s
                <script type="text/javascript">
                    %s
                </script>
                <script src="Chart.js"></script>
                <title>CanvasJS Example</title>
            </head>
            <body>
                %s
            </body>
            </html>
           """

        self._db.create_topic_terms_view()
        css = self.create_wordcloud_for_charts(topic_and_data)
        data_sets = self.create_charts_script(topic_and_data)
        self._write_topic_statistics(topic_and_data)
        whole = wrapper % (css, data_sets, self.create_body_section(len(topic_and_data)))
        f.write(whole)
        f.close()
        open_new_tab(filename)

    def create_topic_data_list(self):
        try:
            print("Generate visualization")
            topic_data_list = []
            topic_to_author_type_num_of_posts_mapping, topic_num_of_authors_distribution_dict = self.generate_datasets_for_visualization()
            topic_keys = list(topic_to_author_type_num_of_posts_mapping.keys())
            for topic in topic_to_author_type_num_of_posts_mapping:
                # for prediction in topic:
                #     if (int(prediction[0]) == topic):
                #         buckets[prediction[1]] = int(prediction[2])
                for key in self._buckets:
                    if key not in topic_to_author_type_num_of_posts_mapping[topic]:
                        topic_to_author_type_num_of_posts_mapping[topic][key] = 0
                    if key not in topic_num_of_authors_distribution_dict[topic]:
                        topic_num_of_authors_distribution_dict[topic][key] = 0

                keys = list(topic_to_author_type_num_of_posts_mapping[topic].keys())
                sorted_keys = sorted(keys, reverse=True)
                labeles = sorted_keys
                number_of_posts_in_topic_distribution_dataset = []
                for key in sorted_keys:
                    number_of_posts_in_topic_distribution_dataset.append(
                        (topic_to_author_type_num_of_posts_mapping[topic])[key])
                number_of_authors_in_topic_distribution_dataset = []
                for key in sorted_keys:
                    number_of_authors_in_topic_distribution_dataset.append(
                        (topic_num_of_authors_distribution_dict[topic])[key])
                data = {}
                color_list = self.red_to_green_colors(10)
                back_ground_color_authors = []
                back_ground_color_posts = []

                data['datasets'] = [
                    {'data': number_of_posts_in_topic_distribution_dataset, 'backgroundColor': color_list},
                    {'data': number_of_authors_in_topic_distribution_dataset, 'backgroundColor': color_list}
                ]
                topic_data_list.append((str(topic), data, labeles))
            self.create_html_file(topic_data_list)
        except IOError as error:
            print("IOError:" + repr(error))

        pass

    def create_topic_wordcloud_img(self, topic, frec):
        # Create wordcloud img for the topic using words frequencies e.g:
        # frec = [('hi', 0.01), ('bye', 0.5), ('wow', 0.3), ('dfg', 0.1), ('efef', 0.9)]

        import wordcloud
        # if you have problem with config - this is it
        # font = "data/output/topic_distribution_visualization/Mukadimah.ttf"
        frec = self._reverse_artbic_words(frec)
        wordcloud = wordcloud.WordCloud(prefer_horizontal=1, ranks_only=True,
                                        background_color='white',
                                        mask=imread('topic_distribution_visualization/red-circle.png')).fit_words(frec)
        picture_path = '%stopic_wordcloud/topic%s.png' % (self._viz_output_path, topic)
        wordcloud.to_file(picture_path)
        pass

    def _reverse_artbic_words(self, frec):
        result = {}
        for word, frequency in frec:
            try:
                if detect(word) == 'ar' or detect(word) == 'he':
                    result[word[::-1]] = frequency
                else:
                    result[word] = frequency
            except Exception as e:
                print(e)
                result[word] = frequency
        return result

    def create_wordcloud_for_charts(self, topic_and_data):
        # create the css section for the visualization of each graph

        css = """<style>
                    body > div {
                    display: inline-block;
                    }
        """
        i = 1
        topic_word_probability_dict = self._db.get_words_with_highest_probability()
        for topic in topic_and_data:
            self.create_topic_wordcloud_img(topic[0], topic_word_probability_dict[topic[0]])
            temp = """
                    #canvas-holder%s {
                        background-image: url("topic_wordcloud/topic%s.png");
                        background-size: 45%s 45%s;
                        background-position: 50%s 60%s;
                        background-repeat: no-repeat;
                    }

            """
            css += temp % (str(i), str(topic[0]), '%', '%', '%', '%')
            i += 1
        css += "</style>"
        return css

    def generate_datasets_for_visualization(self):
        if not self._db.is_post_topic_mapping_table_exist() or not self._db.is_topics_table_exist():
            logging.error("TABLES: post_topic_mapping_table or topics tables DON'T EXIST. APP CONNOT RUN! ABORTING...")
            raise IOError("TABLES: post_topic_mapping_table or topics tables DON'T EXIST. APP CONNOT RUN! ABORTING...")

        author_classification = self.load_authors_classification()
        topic_to_author_type_mapping, topic_authors_distribution_dict = self.generate_topic_to_author_type_mapping(
            author_classification)
        return topic_to_author_type_mapping, topic_authors_distribution_dict

    def load_authors_classification(self):
        '''
        :return: a mapping of author_name -> author badness score (1 = 100% bad, 0 = 100% good)
        '''
        author_classification = {}
        if self._read_predictions_from_db == True and self._include_unlabeled_predictions:
            author_classification = self._fill_author_classification_from_table()
        elif self._include_unlabeled_predictions:  # read from CSV file
            author_classification = self._fill_author_classification_from_csv_file()

        if self._include_labeled_authors_in_visualization:
            labeled_authors = self._db.get_labeled_authors_by_domain(self._domain, self._targeted_class_field_name)
            # reversed_optional_classes_dict = {value: key for key, value in self._optional_classes.iteritems()}
            # bad_type_class_name = reversed_optional_classes_dict[1]
            for labeled_author in labeled_authors:
                author_name = labeled_author[0]
                author_type = labeled_author[1]
                # if author_type == bad_type_class_name:
                #     prediction = 0
                # else:
                #     prediction = 1
                prediction = self._optional_classes[author_type]
                self._fill_author_classification(author_name, author_type, prediction, author_classification)

        return author_classification

    def generate_topic_to_author_type_mapping(self, author_classification):
        '''
        :return: a mapping of <topic_id> -> <author type> -> <number of posts posted by authors of this type in this topic>
        '''
        topic_post_distribution = {}
        topic_authors_distribution_dict = {}
        topic_to_author_mapping = self._db.get_topic_to_author_mapping(self._target_author_field)
        j = 1
        for topic_id in topic_to_author_mapping:
            msg = "\nTopic [{}/{}]".format(str(j), str(len(topic_to_author_mapping)))
            print(msg)
            j += 1
            # topic_name = 'topic' + str(topic_id)
            if topic_id not in topic_post_distribution:
                topic_post_distribution[topic_id] = {}
            if topic_id not in topic_authors_distribution_dict:
                topic_authors_distribution_dict[topic_id] = {}
            i = 1
            for author in topic_to_author_mapping[topic_id]:
                msg = "\rMapping Author: [{}/{}]".format(str(i), str(len(topic_to_author_mapping[topic_id])))
                print(msg, end="")
                i += 1
                category = self.get_author_category(author, author_classification)
                if category is None:
                    continue
                if category not in topic_post_distribution[topic_id]:
                    topic_post_distribution[topic_id][category] = 0
                if category not in topic_authors_distribution_dict[topic_id]:
                    topic_authors_distribution_dict[topic_id][category] = 0

                topic_post_distribution[topic_id][category] += topic_to_author_mapping[topic_id][author]
                topic_authors_distribution_dict[topic_id][category] += 1
        return topic_post_distribution, topic_authors_distribution_dict

    def get_bucket(self, prediction):
        for bucket in self._buckets:
            if prediction < float(bucket):
                return float(bucket)
        return float(1)

    def get_author_category(self, author, author_classification):
        if author in author_classification:
            category = author_classification[author]
        else:
            category = None
        return category

    def _fill_author_classification(self, author_name, predicted, prediction, author_classification):
        good_label = None
        bad_label = None
        for label, num in  self._optional_classes.items():
            if num == 0:
                good_label = label
            elif num == 1:
                bad_label = label

        if (good_label in predicted and prediction >= 0.5):
            prediction = 1 - prediction
        if (bad_label in predicted and prediction <= 0.5):
            prediction = 1 - prediction
        author_classification[author_name] = self.get_bucket(prediction)

    def _fill_author_classification_from_table(self):
        author_classification = {}
        is_predictions_table_exist = self._db.is_table_exist('unlabeled_predictions')
        if is_predictions_table_exist == False:
            message = "TABLE: unlabeled_predictions DON'T EXIST. APP CONNOT RUN! ABORTING..."
            logging.error(message)
            raise IOError(message)
        cursor = self._db.get_unlabeled_predictions()
        unlabeled_predictions_generator = self._db.result_iter(cursor)
        for tuple in unlabeled_predictions_generator:
            author_name = tuple[1]
            predicted = tuple[2]
            prediction = tuple[3]
            self._fill_author_classification(author_name, predicted, prediction, author_classification)

        return author_classification

    def _fill_author_classification_from_csv_file(self):
        author_classification = {}
        print("Load authors predications CSV")
        prediction_csv = pd.DataFrame.from_csv(self.prediction_csv_path)
        prediction_csv.reset_index(inplace=True)
        for i, row in enumerate(prediction_csv.itertuples(index=False)):
            print(row)
            author_name = row[0]
            predicted = row[1]
            prediction = float(row[2])
            self._fill_author_classification(author_name, predicted, prediction, author_classification)
        return author_classification

    def _write_topic_statistics(self, topic_and_data):
        dfs = []
        for i, topic_tuple in enumerate(topic_and_data):
            topic_number = int(topic_tuple[0])
            buckets = topic_tuple[2]
            #buckets = buckets
            #topic_buckets = ["topic_" + str(i) + "_" + str(bucket) for bucket in buckets]
            #topic_buckets = ["topic_{0}_{1}".format(i + 1, bucket) for bucket in buckets]

            df = pd.DataFrame()

            posts = topic_tuple[1]['datasets'][0]['data']
            authors = topic_tuple[1]['datasets'][1]['data']

            post_author_ratios = []
            for j, post in enumerate(posts):
                authors_in_bucket = authors[j]
                posts_in_bucket = posts[j]

                if authors_in_bucket != 0:
                    ratio = posts_in_bucket / float(authors_in_bucket)
                else:
                    ratio = -1.0
                post_author_ratios.append(ratio)

            df['authors'] = authors
            df['posts'] = posts
            df['post_author_ratio'] = post_author_ratios
            df['buckets'] = buckets
            df['topic_number'] = [topic_number] * len(buckets)

            df.set_index(['topic_number', 'buckets'], inplace=True)
            # reverse dataframe
            df = df[::-1]

            dfs.append(df)
        if len(dfs) > 0:
            topic_df = pd.concat(dfs)
            topic_df.to_csv(self._viz_output_path + "topic_statistics.csv")

