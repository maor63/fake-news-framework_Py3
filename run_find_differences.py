import json
from pprint import pprint
from datetime import datetime, timedelta
from email.utils import parsedate_tz

def read_file(path):
        if path is not '':
            with open(path) as data_file:
                posts = json.load(data_file)
                print(("FileName: " + path + "----------------------"))
                num_of_posts = len(posts)
                print(("Number of given posts in file: " + str(num_of_posts)))
                return posts

def read_files(path1, path2):
    file_1_posts = read_file(path1)
    file_2_posts = read_file(path2)
    return (file_1_posts, file_2_posts)

def populate_dictionary(posts):
    if posts is not None:
        for post in posts:
            id = post['id']
            occurences = dictionary.get(id)
            if occurences == None:
                dictionary[id] = 1
            else:
                dictionary[id] += 1

def to_datetime(datestring):
    time_tuple = parsedate_tz(datestring.strip())
    dt = datetime(*time_tuple[:6])
    return dt - timedelta(seconds=time_tuple[-1])

def find_newest_and_oldest_dates(path, posts):
    if posts is not None:
        print(("FileName: " + path + "----------------------"))

        dates = fill_post_dates(posts)

        formated_dates = list(dates.values())

        max_date = str(max(formated_dates))
        print(("The neweset date is: " + max_date))

        min_date = str(min(formated_dates))
        print(("The oldeset date is: " + min_date))

def fill_post_dates(posts):
    if posts is not None:
        dates = {}
        for post in posts:
            str_date = post['created_at']
            post_date = to_datetime(str_date)
            occurences = dates.get(str_date)
            if occurences == None:
                dates[str_date] = post_date + timedelta(hours=3)
        return dates

if __name__ == '__main__':

    try:

        dictionary = {}

        #path1 = "data\docs\VirtualTV_201604260930.json"
        #path2 = "data\docs\Virtualtv_201604260935.json"
        #path1 = "data\docs\Online_TV_201604261417.json"
        #path2 = "data\docs\Online_tv_201604261423.json"
        #path1 = "data\docs\Internet_TV_201604261443.json"
        #path2 = "data\docs\Internet_tv_201604261444.json"
        #path1 = "data\docs\SmartTV_201604261511.json"
        #path2 = "data\docs\Smarttv_201604261512.json"
        #path1 = "data\docs\Chernobyl_disaster_201604261548.json"
        #path1 = "data\docs\Chernobyl_disaster_201604261623.json"
        #path2 = ""

        #path1 = "data\docs\Online_TV_201604270952_Popular_100.json"
        #path2 = "data\docs\Online_TV_201604271024_Recent_100.json"

       # path1 = "data\docs\Online_TV_201604271024_Recent_100.json"
       # path2 = "data\docs\Online_TV_201604271031_Mixed_100.json"

        #path1 = "data\docs\Online_TV_201604271031_Mixed_100.json"
        #path2 = "data\docs\Online_TV_201604271037_100.json"

        path1 = "data\docs\Online_TV_201605011141_my_post_is_not_found.json"
        path2 = "data\docs\watch_Game_of_Thrones_online_TV_201605011506.json"

        file_1_posts, file_2_posts = read_files(path1, path2)

        find_newest_and_oldest_dates(path1, file_1_posts)
        find_newest_and_oldest_dates(path2, file_2_posts)

        populate_dictionary(file_1_posts)
        populate_dictionary(file_2_posts)

        number_of_different_posts = len(dictionary)
        print(("Number of different posts: " + str(number_of_different_posts)))

        for keys,values in list(dictionary.items()):
            print(keys)
            print(values)
    except IOError as e:
        print(("No such file or directory:" + e.filename))
    except ValueError as e:
        print("Please check the given file. Maybe is not valid JSON file")