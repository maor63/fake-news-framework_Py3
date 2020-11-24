
import http.cookiejar
import datetime
import json
import re
import sys
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
import time

from pyquery import PyQuery

from .. import models


class TweetManager:

    def __init__(self):
        pass

    @staticmethod
    def getTweets(tweetCriteria, receiveBuffer=None, bufferLength=100, proxy=None, lang='en'):
        refreshCursor = ''

        results = []
        resultsAux = []
        cookieJar = http.cookiejar.CookieJar()

        if hasattr(tweetCriteria, 'username') and (
                tweetCriteria.username.startswith("\'") or tweetCriteria.username.startswith("\"")) and (
                tweetCriteria.username.endswith("\'") or tweetCriteria.username.endswith("\"")):
            tweetCriteria.username = tweetCriteria.username[1:-1]

        active = True

        seen_tweets = set()
        continued = 0

        while active:
            # print("\rtweets retrieved: {0}, skipped: {1}".format(len(seen_tweets), continued), end='')
            json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy, lang)
            if len(json['items_html'].strip()) == 0:
                break

            # refreshCursor = json['min_position']

            min_position = json['min_position']
            refreshCursor = min_position.replace("+", "%2B")

            scrapedTweets = PyQuery(json['items_html'])
            # Remove incomplete tweets withheld by Twitter Guidelines
            scrapedTweets.remove('div.withheld-tweet')
            tweets = scrapedTweets('div.js-stream-tweet')

            if len(tweets) == 0:
                break
            tweet_ids = []
            for tweetHTML in tweets:
                tweetPQ = PyQuery(tweetHTML)
                tweet = models.Tweet()

                usernameTweet = tweetPQ("span:first.username.u-dir b").text()
                txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'))
                retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr(
                    "data-tweet-stat-count").replace(",", ""))
                favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr(
                    "data-tweet-stat-count").replace(",", ""))
                dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"))
                id = tweetPQ.attr("data-tweet-id")
                permalink = tweetPQ.attr("data-permalink-path")

                tweet_ids.append(id)

                geo = ''
                geoSpan = tweetPQ('span.Tweet-geo')
                if len(geoSpan) > 0:
                    geo = geoSpan.attr('title')

                tweet.id = id
                tweet.permalink = 'https://twitter.com' + permalink
                tweet.username = usernameTweet
                tweet.text = txt
                tweet.date = datetime.datetime.fromtimestamp(dateSec)
                tweet.retweets = retweets
                tweet.favorites = favorites
                tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))
                tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(tweet.text))
                tweet.geo = geo

                if tweet.id not in seen_tweets:
                    results.append(tweet)
                    resultsAux.append(tweet)
                    seen_tweets.add(tweet.id)
                    continued = 0
                else:
                    continued += 1
                    if continued > 10:
                        active = False
                        break
                    continue

                if receiveBuffer and len(resultsAux) >= bufferLength:
                    receiveBuffer(resultsAux)
                    resultsAux = []

                if tweetCriteria.maxTweets > 0 and len(results) >= tweetCriteria.maxTweets:
                    active = False
                    break

            # refreshCursor = 'TWEET-{0}-{1}'.format(tweet_ids[0], tweet_ids[-1])
        if receiveBuffer and len(resultsAux) > 0:
            receiveBuffer(resultsAux)
        # print()
        return results

    @staticmethod
    def getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy, lang='en'):
        url = "https://twitter.com/i/search/timeline?l={}&f=tweets&q=%s&src=typd&max_position=%s".format(lang)

        urlGetData = ''

        if hasattr(tweetCriteria, 'username'):
            urlGetData += ' from:' + tweetCriteria.username

        if hasattr(tweetCriteria, 'querySearch'):
            urlGetData += ' ' + tweetCriteria.querySearch

        if hasattr(tweetCriteria, 'near'):
            urlGetData += "&near:" + tweetCriteria.near + " within:" + tweetCriteria.within

        if hasattr(tweetCriteria, 'since'):
            urlGetData += ' since:' + tweetCriteria.since

        if hasattr(tweetCriteria, 'until'):
            urlGetData += ' until:' + tweetCriteria.until

        if hasattr(tweetCriteria, 'topTweets'):
            if tweetCriteria.topTweets:
                url = "https://twitter.com/i/search/timeline?l={}&q=%s&src=typd&max_position=%s".format(lang)

        url = url % (urllib.parse.quote(urlGetData), urllib.parse.quote(refreshCursor))

        headers = [
            ('Host', "twitter.com"),
            ('User-Agent', "Googlebot/2.1 (+http://www.google.com/bot.html)"),
            # ('User-Agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"),
            ('Accept', "application/json, text/javascript, */*; q=0.01"),
            ('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),
            ('X-Requested-With', "XMLHttpRequest"),
            ('Referer', url),
            ('Connection', "keep-alive")
        ]

        for i in range(15):
            if proxy:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}),
                                              urllib.request.HTTPCookieProcessor(cookieJar))
            else:
                opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookieJar))
            opener.addheaders = headers

            try:
                response = opener.open(url)
                jsonResponse = response.read()
                dataJson = json.loads(jsonResponse)
                return dataJson
            except:
                print ('\r reach rate limit sleeps for 30 sec, count {}'.format(i), end='')
                time.sleep(30)
        print()
        print ("Twitter weird response. Try to see on browser: https://twitter.com/search?q=%s&src=typd" % urllib.parse.quote(
            urlGetData))
        sys.exit()
        return


