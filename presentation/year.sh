#!/bin/bash -l 

YEAR=${1:-2023}

ls -l *${YEAR}*.txt | grep -v TALK 
