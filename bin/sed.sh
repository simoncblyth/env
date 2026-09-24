#!/usr/bin/env bash

usage(){ cat << EOU

    Usage: sed.sh <file> [start_line] [end_line]

    Examples:
        sed.sh DsG4OpAbsReemit/gemini.txt 100 | pbcopy
        sed.sh DsG4OpAbsReemit/gemini.txt 100 150 | pbcopy

EOU
}

if [ -z "$1" ]; then
    usage
    exit 1
fi

file="$1"       # required path argument
from="${2:-1}"  # default from "1" start 
to="${3:-\$}"   # default to "$" end

sed -n "${from},${to}p" "$file"


