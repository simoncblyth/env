#!/usr/bin/env python
"""

* http://pymotw.com/2/profile/#module-pstats
* http://pymotw.com/2/profile/

"""
import os, sys, pstats

def present_cprofile( path ):
    s = pstats.Stats(path)
    s.strip_dirs()               # Clean up filenames for the report
    s.sort_stats('cumulative')   # Sort the statistics by the cumulative time spent in the function
    s.print_stats()   

if __name__ == '__main__':
    present_cprofile( sys.argv[1] )




