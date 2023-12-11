#!/bin/bash -l 
usage(){ cat << EOU
grep.sh
=========

::

    ~/env/presentation/grep.sh Fastener

Greps .txt files in the presentation folder excluding _TALK.txt 
and changes the .txt path listing into URLs 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

#host="http://localhost"
host="open http://localhost"
HOST="${HOST:-$host}"
pattern=${1:-Fastener}

grep $pattern -r --include=\*.txt --exclude=\*_TALK.txt $PWD \
    | perl -pe "s,$HOME/env,$HOST/env," - \
    | perl -pe "s,.txt,.html," -


