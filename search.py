from __future__ import print_function

import argparse
import csv
import glob

import cv2

from coverdescriptor import CoverDescriptor
from covermatcher import CoverMatcher

argparser = argparse.ArgumentParser()
argparser.add_argument('-d', '--db', required=True, help='Path to the book database.')
argparser.add_argument('-c', '--covers', required=True, help='Path to the directory that contains our book covers.')
argparser.add_argument('-q', '--query', required=True, help='Path to the query book cover.')
argparser.add_argument('-s', '--sift', required=False, type=int, default=0, help='Whether or not SIFT should be used.')
arguments = vars(argparser.parse_args())

db = dict()

for line in csv.reader(open(arguments['db'])):
    db[line[0]] = line[1:]

use_sift = arguments['sift'] > 0
use_hamming = arguments['sift'] == 0
ratio = 0.7
min_matches = 40

if use_sift:
    min_matches = 50

cover_descriptor = CoverDescriptor(use_sift=use_sift)
cover_matcher = CoverMatcher(cover_descriptor, glob.glob(arguments['covers'] + '/*.png'), ratio=ratio,
                             min_matches=min_matches, use_hamming=use_hamming)

query_image = cv2.imread(arguments['query'])
gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
(query_keypoints, query_descriptors) = cover_descriptor.describe(gray)

results = cover_matcher.search(query_keypoints, query_descriptors)

cv2.imshow('Query', query_image)

if len(results) == 0:
    print('I could not find a match for that cover!')
    cv2.waitKey(0)
else:
    for (i, (score, cover_path)) in enumerate(results):
        (author, title) = db[cover_path[cover_path.rfind('/') + 1:]]
        print('{}. {:2f}% : {} - {}'.format(i + 1, score * 100, author, title))

        result = cv2.imread(cover_path)
        cv2.imshow('Result', result)
        cv2.waitKey(0)
