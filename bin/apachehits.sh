#!/bin/bash -l

usage(){ cat << EOU

Usage::

   apachehits.sh    # todays hourly hit counts
   apachehits.sh DAY=20/Jun/2014    
   # counts from a particular day, using date formatting used in access_log 

EOU
}


apache-
python-

cmd="$* LOG=$(apache-logdir)/access_log apachehits.py"
echo $cmd
eval $cmd



