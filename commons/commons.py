# Created by aviade      
# Time: 03/05/2016 09:00

import datetime
import string
from datetime import timedelta
import time
import urllib.request, urllib.error, urllib.parse
import unicodedata
import uuid
import sys
import logging
import os

import re

import numpy
from decimal import Decimal
from nltk.stem.snowball import GermanStemmer, EnglishStemmer
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.simple import SpaceTokenizer
from networkx import Graph, DiGraph
from nltk.corpus import stopwords


def str_to_date(datestring, formate="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.strptime(datestring, formate)


date = str_to_date


def date_to_str(datetimeObj, formate="%Y-%m-%d %H:%M:%S"):
    return datetimeObj.strftime(formate)


def cleaner(dirtyStr):
    # @review: cleaner of what?
    # @todo: refactor rename

    # from old clean data:
    afterOldCleandData = ""

    afterOldCleandData = dirtyStr.replace("<![CDATA[", "").replace("]]>", ""). \
        replace("\n", "").replace("\r\n", "").replace("\r", "").replace("\t", "")
    dashes = afterOldCleandData.find('#')
    if (dashes != -1):
        afterOldCleandData = afterOldCleandData[:dashes]
    if afterOldCleandData.endswith('/'):
        afterOldCleandData = afterOldCleandData[:-1]
    # from old clean url code:
    afterOldCleanUrl = afterOldCleandData.strip('\r\n\t')
    stripedStr = afterOldCleanUrl.strip("\r\n ")
    if stripedStr.startswith('"') or stripedStr.endswith('"'):
        stripedStr = stripedStr[1:-1]
    stripedStr = stripedStr.strip("(")
    stripedStr = stripedStr.strip(")")

    cleanStr = stripedStr.lstrip().rstrip()
    # cleanStr=cleanStr[:cleanStr.find('#')]
    return cleanStr


def normalize_url(url):
    '''
    Cleans the url and adjust it to the required encoding conventions:
    * The url is saved quoted
    * the safe characters remain unquoted
    '''
    if ('%' in url):

        try:
            url = urllib.parse.unquote(url.encode('utf-8')).decode('utf8')
        except:
            url = urllib.parse.unquote(url.encode('utf-8').decode('utf8'))
            # orgurl = urllib2.unquote(orgurl.encode('ascii')).decode('utf8')
    else:
        url = url.encode('utf-8').decode('utf8')
    url = cleaner(url)

    normalized_url = urllib.parse.quote(url.encode('utf-8'), safe="://!*'();:@&=+$,/?%#[]")

    return normalized_url


def createunicodedata(data):
    '''
    :param data: A string \ unicode object
    :return: a string representation of the given data in which all compatibility characters are replaces with their
    equivalents. For more into see: https://docs.python.org/2/library/unicodedata.html#unicodedata.normalize
    '''
    return unicodedata.normalize('NFKD', str(data)).encode('ascii', 'ignore')


def cleanForAuthor(dirtyAuthorStr):
    sentToCleaner = dirtyAuthorStr.replace("-", "").replace(".", "")  # .lower()
    return cleaner(sentToCleaner)


def str_to_date(datestring, formate="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.strptime(datestring, formate)


def convert_str_to_unicode_datetime(str_date):
    str_date = str(str_date)
    # unicode_str_date = unicodedata.normalize('NFKD', str_date).encode('ascii', 'ignore')
    date = str_to_date(str_date)
    return date


def get_current_time_as_string():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def compute_author_guid_by_author_name(author_name):
    from configuration.config_class import getConfig
    configInst = getConfig()
    prefix_osn_url = configInst.eval("DEFAULT", "social_network_url")
    author_url = prefix_osn_url + author_name

    # bytes = get_bytes(author_url)

    class NULL_NAMESPACE:
        bytes = b''
    author_guid = uuid.uuid3(NULL_NAMESPACE, author_url)
    str_author_guid = str(str(author_guid))

    return str_author_guid


def generate_random_guid():
    guid = uuid.uuid4()
    str_guid = str(guid)
    return str_guid


def compute_post_guid(post_url, author_name, str_publication_date):
    author_guid = compute_author_guid_by_author_name(author_name)
    publication_date = datetime.datetime.strptime(str_publication_date, '%Y-%m-%d %H:%M:%S')

    # adding two hours according to Henrik
    german_publication_date = publication_date + timedelta(hours=2)

    epoch = datetime.datetime.utcfromtimestamp(0)
    german_publication_date_in_milliseconds = int((german_publication_date - epoch).total_seconds() * 1000)

    str_german_publication_date_in_milliseconds = str(german_publication_date_in_milliseconds)

    url = post_url + "#" + author_guid + "#" + str_german_publication_date_in_milliseconds

    class NULL_NAMESPACE:
        bytes = b''

    post_guid = uuid.uuid3(NULL_NAMESPACE, url)
    str_author_guid = str(str(post_guid))
    return str_author_guid
    '''

    long_publication_date = convert_date_to_long(publicationDate);
    String strLongPublicationDate = longPublicationDate.toString();

    System.out.println("Post url = " + postUrl);
    System.out.println("AuthorGuid = " + authorGuid);
    System.out.println("strLongPublicationDate = " + strLongPublicationDate);
    String url = postUrl + "#" + authorGuid + "#" + strLongPublicationDate;


    byte[] urlBytes = url.getBytes();
    UUID postGuid = UUID.nameUUIDFromBytes(urlBytes);
    String strPostGuid = postGuid.toString();

    System.out.println("Post: java.util.UUID .nameUUIDFromBytes((<URL> + \"#\" + <AUTHOR_GUID> + \"#\" + <PubDateAsLong>).getBytes())");
    System.out.println("POST GUID: "+ strPostGuid);
    '''


def convert_date_to_long(str_date):
    return 0


def split_into_equal_chunks(elements, num_of_chunks):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(elements), num_of_chunks):
        yield elements[i:i + num_of_chunks]


def count_down_time(seconds_to_wait):
    for i in range(seconds_to_wait, 0, -1):
        time.sleep(1)
        msg = "\rCount Down {0}".format(str(i))
        print(msg, end="")
        # sys.stdout.flush()


def print_list_ids(items):
    ids = ",".join([str(item) for item in items])
    print(ids)
    logging.info(ids)


def create_ids_from_config_file(source_ids_for_fast_running):
    author_ids = []
    source_ids = source_ids_for_fast_running.split(",")

    for source_id in source_ids:
        source_id = source_id.strip()
        author_id = int(source_id)
        author_ids.append(author_id)
    return author_ids


def retreive_valid_k(k, author_type_class_series):
    series_length = len(author_type_class_series)
    if series_length < k:
        return series_length
    else:
        return k


def retreive_labeled_authors_dataframe(targeted_class_name, dataframe):
    labeled_dataframe = dataframe.loc[dataframe[targeted_class_name].notnull()]
    return labeled_dataframe


def move_existing_file_to_backup(file_name, path, backup_path):
    full_path_output_file = path + file_name
    if os.path.isfile(full_path_output_file):
        full_path_backup_output_file = backup_path + file_name
        if os.path.isfile(full_path_backup_output_file):
            os.remove(full_path_backup_output_file)
        os.rename(full_path_output_file, full_path_backup_output_file)


def get_boolean_value(boolean_value):
    if boolean_value is None:
        return None
    elif boolean_value is True:
        return 1
    return 0


def convert_epoch_timestamp_to_datetime(str_epoch_timestamp):
    int_post_creation_date = int(str_epoch_timestamp)
    str_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int_post_creation_date / 1000.0))
    datetime = str_to_date(str_datetime)
    return datetime, str(str_datetime)


# def create_author_dictionary(authors):
#     """
#     Converts a list of Authors objects into a dictionary.
#     The dictionary's key is Author OSN Id
#     The value is a tuple containing Author OSN Id, Author Type, Sub Type and Author GUID
#     :param authors:
#     :return: dictionary of authors
#     """
#     author_dictionary = {}
#     for author in authors:
#         author_osn_id = int(author.author_osn_id)
#         author_dictionary[author_osn_id] = (author_osn_id, author.author_type, author.author_sub_type,
#                                             author.author_guid)
#     #TODO remove redudant author_osn_id in value of the dictionary
#     return author_dictionary


def create_author_dictionary(self, authors):
    """
    Converts a list of Authors objects into a dictionary.
    The dictionary's key is author_guid
    The value is a tuple containing Author Type, Sub Type
    :param authors:
    :return: dictionary of authors
    """
    author_dictionary = {}
    for author in authors:
        author_guid = author.author_guid
        # author_osn_id = getattr(author, "author_osn_id")
        tuple = ()
        # tuple = tuple + (author_osn_id,)

        for targeted_class in self._targeted_classes:
            targeted_class_value = getattr(author, targeted_class)

            tuple = tuple + (targeted_class_value,)

        author_dictionary[author_guid] = tuple
    return author_dictionary


def replace_nominal_class_to_numeric(dataframe, optional_classes):
    num_of_class = len(optional_classes)
    for i in range(num_of_class):
        class_name = optional_classes[i]
        dataframe = dataframe.replace(to_replace=class_name, value=i)
    return dataframe


def generate_tweet_url(tweet_id, tweet_author_name):
    '''
    :return: the URL of the retweeted tweet
    '''
    url = "https://twitter.com/{0}/status/{1}".format(tweet_author_name, tweet_id)
    return url


def extract_tweet_publiction_date(tweet_creation_time):
    '''
    :param tweet_creation_time: the time in which the tweet was published
    :return: the publication date of the tweet as a string.
    The time is CEST time.
    the structure of the time signature: YYYY-MM-DD HH:MM:SS
    '''
    utc_repr = datetime.datetime.strptime(tweet_creation_time, '%a %b %d %H:%M:%S +0000 %Y')
    cest_repr = utc_repr + timedelta(hours=2)
    return str(cest_repr)


def clean_content_to_set_of_words(stopwords_file, content, stemmerLanguage):
    """Content is assumed to be a unicode object in utf-8."""
    stemmers = set_stemmer(stemmerLanguage)
    p = "[A-Za-z']+".decode('utf-8')

    # c = re.compile(p)
    content = content.lower()
    content = clean_word(content)
    # stop_words = set_of_stopwords(stopwords_file)
    stop_words = set(stopwords.words('english'))
    tokens = [_ for _ in content.split(" ") if len(_) > 0]
    hrefs = []
    words = []
    for token in tokens:
        if token.startswith("http://") or token.startswith("https://"):
            hrefs.append(token)
        else:
            words.append(token)

    words = tokens
    # content = " ".join(words)
    # words = c.findall(content)
    words = re.sub(r'^https?:\/\/.*[\r\n]*', '', ' '.join(words), flags=re.MULTILINE).split()  # Remove urls.

    words = [word for word in words if word not in stop_words]

    words = [stem(word, stemmers) for word in words]

    words = [word for word in words if 2 <= len(word) <= 25]

    return words


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def set_stemmer(stemmer_language):
    if (stemmer_language == "GER"):
        stemmers = GermanStemmer()
    else:
        stemmers = EnglishStemmer()
    return stemmers


def set_of_stopwords(filename):
    if not os.path.exists(filename):
        return set()
    with open(filename) as f:
        return set(line.strip() for line in f.readlines())
        # set((line.strip(' ') for line in f.readlines()))


def stem(word, stemmer):
    word = stemmer.stem(word)

    return word

def calc_ngrams(words, minngram, maxngram):
    """For a given content string, calculate the set of ngrams."""

    vectorizer = CountVectorizer(ngram_range=(minngram, maxngram))
    analyzer = vectorizer.build_analyzer()
    ngrams_set = analyzer(words)
    ngrams_set = set(ngrams_set)

    return list(("".join(ngram) for ngram in ngrams_set))


def goodbye_string(obj, encoding='utf-8'):
    """For objects leaving the system. Returns an str instance."""
    # @review: why "leaving the system"? what is the purpose of this function?
    if isinstance(obj, str):
        if isinstance(obj, str):
            obj.encode(encoding, 'ignore')
    return obj


def get_words_by_content(content):
    words = []
    tokenizer = SpaceTokenizer()
    words += tokenizer.tokenize(content)
    ##words = list(set(words))
    # words = frozenset(words)
    return words


def euclidean_distance(vector_a, vector_b):
    sub = distance.euclidean(vector_a, vector_b)
    return sub


def cosine_similarity(vector_a, vector_b):
    numerator = sum(a * b for a, b in zip(vector_a, vector_b))
    denominator = _square_rooted(vector_a) * _square_rooted(vector_b)
    if denominator == 0:  # check if this is right!
        return 0
    return round(numerator / float(denominator), 3)


def _square_rooted(x):
    return round(numpy.sqrt(sum([a * a for a in x])), 3)


def minkowski_distance(vector_a, vector_b):
    return _nth_root(sum(pow(abs(a - b), 3) for a, b in zip(vector_a, vector_b)), 3)


def _nth_root(value, n_root):
    root_value = 1 / float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)


def manhattan_distance(vector_a, vector_b):
    return sum(abs(a - b) for a, b in zip(vector_a, vector_b))


def jaccard_index(vector_a, vector_b):
    set_1 = set(vector_a)
    set_2 = set(vector_b)
    union_size = len(set_1 | set_2)
    if union_size:
        return (len(set_1 & set_2)) / float(union_size)
    else:
        return 0.0
    # n = len(set_1.intersection(set_2))
    # return n / float(len(set_1) + len(set_2) - n)


def clean_word(word):
    return re.sub('[^a-zA-Z ]+', '', word)


# TODO: refactor code in link_prediction_feature_extractor
def create_targeted_graph(graph_type):
    if (graph_type == 'undirected'):
        graph = Graph()
    elif (graph_type == 'directed'):
        graph = DiGraph()
    else:
        print("Graph type has to be directed or undirected")
        return None

    return graph


def fill_edges_to_graph(graph, tuples):
    edges = []
    for tuple in tuples:
        source_author_guid = tuple[0]
        dest_author_guid = tuple[1]
        weight = tuple[2]

        edges.append((source_author_guid, dest_author_guid, {"weight": weight}))

    graph.add_edges_from(edges)


def distance_calculator(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    # origin = (48.1372, 11.5756)  # Munich
    # destination = (52.5186, 13.4083)  # Berlin
    # round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # approximately radius of earth

    dlon = numpy.math.radians(lon2 - lon1)
    dlat = numpy.math.radians(lat2 - lat1)
    a = (numpy.math.sin(dlat / 2) * numpy.math.sin(dlat / 2) +
         numpy.math.cos(numpy.math.radians(lat1)) * numpy.math.cos(numpy.math.radians(lat2)) *
         numpy.math.sin(dlon / 2) * numpy.math.sin(dlon / 2))
    c = 2 * numpy.math.atan2(numpy.math.sqrt(a), numpy.math.sqrt(1 - a))
    d = radius * c

    return d


def clean_tweet(content, stemmer=None):
    content = content.lower()
    exclude = set(string.punctuation)
    content = ''.join(ch for ch in content if ch not in exclude)

    content = content.replace('&amp;', '&')
    content = content.replace(',', '')
    content = content.replace('!', '')
    content = content.replace('-', '')
    content = content.replace('.', '')
    if 'http' in content or 'www' in content:
        content = re.sub(r'http\S+', '', content)
        content = re.sub(r'www\S+', '', content)
    if stemmer:
        stemmers = set_stemmer(stemmer)
        content = ' '.join([stem(word, stemmers) for word in content.split()])
    return content


def convert_claim_to_post(claim):
    from DB.schema_definition import Post
    post = Post()
    post.post_id = claim.claim_id
    post.content = claim.title
    post.description = claim.description
    post.url = claim.url
    post.date = claim.verdict_date
    post.domain = 'Claim'
    post.author = 'no author'
    post.author_guid = 'no author'
    post.guid = compute_post_guid(claim.url, post.author, date_to_str(post.date))
    post.post_osn_guid = post.guid
    post.tags = claim.keywords
    post.post_type = claim.verdict
    return post

def clean_words_from_stopwords(stopwords, words):
    return [word for word in words if word not in stopwords]


def clean_content_by_nltk_stopwords(topic_content):
    stopWords = set(stopwords.words('english'))
    topic_content = ' '.join(clean_words_from_stopwords(stopWords, topic_content.split(' ')))
    return topic_content


def remove_punctuation_chars(word):
    new_word = remove_unprintable_chars(word)
    if new_word.isdigit():
        return new_word
    else:
        for x in string.punctuation:
            if x != '-' and x in word:
                new_word = new_word.replace(x, '')
        return new_word


def clean_claim_description(description, remove_stop_words):
    description = " ".join(description.split())
    description = description.replace('"', '').replace("'", '')
    description = description.replace('.', '')
    description = re.sub(r'http\S+', '', description)
    description = description.replace('&amp;', '&')

    clean_words = []
    for word in description.split():
        new_word = remove_punctuation_chars(word)
        clean_words.append(new_word)

    description = ' '.join(clean_words)



    description = clean_tweet(description)
    if remove_stop_words:
        return clean_content_by_nltk_stopwords(description)
    else:
        return description


def remove_unprintable_chars(word):
    new_word = word
    for x in word:
        if x not in string.printable:
            new_word = new_word.replace(x, '')
    return new_word


def calculate_correctly_and_not_correctly_instances(confusion_matrix):
    num_of_correct_instances = 0
    num_of_incorrect_instances = 0

    dimension = confusion_matrix.shape[0]
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                num_of_correct_instances += confusion_matrix[i][j]
            else:
                num_of_incorrect_instances += confusion_matrix[i][j]

    return num_of_correct_instances, num_of_incorrect_instances
