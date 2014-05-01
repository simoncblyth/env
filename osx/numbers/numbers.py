#!/usr/bin/env python
"""
Usage::

    ./numbers.py ~/Desktop/export.csv 

"""
import sys, csv

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as fp:
        rdr = csv.reader(fp, delimiter='|') 
        for i,row in enumerate(rdr):
            print i, row[0]




