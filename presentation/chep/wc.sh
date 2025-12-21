#!/bin/sh
usage(){ cat <<EOU

2026 : The recommended length of an abstract is 100-250 words. $(( (100+250)/2 ))

EOU
}

usage

wc -w *_abstract.txt


