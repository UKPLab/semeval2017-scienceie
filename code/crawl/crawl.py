#!/usr/bin/python
# -*- coding: UTF-8 -*-
import errno
import json
import logging
import os
import requests
import sys

logging.basicConfig(format='%(asctime)s %(message)s', filename=os.path.join(os.path.dirname(__file__), 'crawl.log'), level=logging.WARN)
logging.getLogger().addHandler(logging.StreamHandler())

MAX_RESULTS = 5000		# ~5000 is max
RESULTS_PER_REQUEST = 200	# 200 is the maximum amount per request set by Elsevier


# load API key
with open(os.path.join(os.path.dirname(__file__), 'API_KEY'), 'r') as api_key_file:
    API_KEY = api_key_file.read().strip()

OUTPUT_RAW = os.path.join(os.path.dirname(__file__), 'output', 'raw')
#OUTPUT_TRUNCATED = os.path.join(os.path.dirname(__file__), 'output', 'truncated')


#
# main method that should be called with a subject code: http://api.elsevier.com/content/subject/scidir?httpAccept=text/xml
#
def crawl_subject(subject, year):
    create_dirs(subject)
    piis = getPiis(subject, year)
    for pii,title in piis.iteritems():
        retrieveXml(subject, pii)


#
# create directories
#
def create_dirs(subject):
    try:
        os.makedirs(os.path.join(OUTPUT_RAW, subject))
#        os.makedirs(os.path.join(OUTPUT_TRUNCATED, subject))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


#
# get a list of PIIs to download
#
def getPiis(subject, year):
    url = 'http://api.elsevier.com/content/search/scidir'
    params = {'subscribed': 'true',				# only subscribed articles
#              'oa': 'true',						# additionally, OA articles; DOES NOT WORK! Returns the same results for all subjects!
              'content': 'journals',				# only from journals
              'count': str(min(RESULTS_PER_REQUEST, MAX_RESULTS)),	# number of articles
              'subj': subject,					# http://api.elsevier.com/content/subject/scidir?httpAccept=text/xml
              'query': 'pub-date AFT ' + year + '0101 AND pub-date BEF ' + year + '1231 AND doc-head(article)',	# needs to be an article AND only published after this date ()
#              'query': 'doc-head(article)',		# needs to be an article
              'field': 'dc:title,pii'}			# only return PII (and title for debugging), as this is all info we need to crawl the page
    headers = {'Accept':'application/json',
               'X-ELS-APIKey': API_KEY}

    piis = dict()
    for start in xrange(0, MAX_RESULTS, RESULTS_PER_REQUEST):
        params['start'] = start
        response = requests.get(url, params=params, headers=headers)
        logging.info('Retrieved PIIs (subj=' + subject + '): ' + response.url)
        logging.info('Total results: ' + response.json()['search-results']['opensearch:totalResults'])
        for result in response.json()['search-results']['entry']:
            logging.info('Retrieved PII(' + subject + '): ' + result['pii'] + '\t' + result['dc:title'])
            piis[result['pii']] = result['dc:title']
    return piis


#
# retrieve full XML for a given pii, write to file
#
def retrieveXml(subject, pii):
    url = "http://api.elsevier.com/content/article/pii/" + normalizePii(pii)
    params = {'view': 'full'}
    headers = {'Accept':' text/xml', # use text/plain for (mostly) plain-text
             'X-ELS-APIKey': API_KEY}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        logging.warn('Successfully retrieved article(' + subject + '): ' + url)
    else:
        logging.warn('Could not download article(' + subject + '): ' + url)
        return

    # write raw response xml
    raw = os.path.join(OUTPUT_RAW, subject, normalizePii(pii)) + '.xml'
    if not os.path.isfile(raw):
        with open(raw, 'w') as f:
            f.write(response.content)

#    # write truncated xml
#    truncated = os.path.join(OUTPUT_TRUNCATED, subject, normalizePii(pii)) + '.xml'
#    if not os.path.isfile(truncated):
#        with open(truncated, 'w') as f:
#            begin = response.content.find('<originalText>') + len('<originalText>')
#            end = response.content.find('</originalText>')
#            f.write('<?xml version="1.0"?>')
#            f.write(response.content[begin:end])


#
# "unformats" a given pii to have an identifier usable for retrieval and storage
# ex:  S0010-938X(15)00195-X  ->  S0010938X1500195X
#
def normalizePii(pii):
    return pii.replace('-', '').replace('(', '').replace(')', '')


# crawl given subject for given year
if len(sys.argv) < 3:
    print "Usage: python crawl.py SUBJECT_CODE YEAR"
    sys.exit(0)

subject = sys.argv[1]
year = sys.argv[2]

crawl_subject(subject, year)




