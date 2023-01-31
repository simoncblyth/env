#!/bin/bash -l 

YEAR=${1:-2022}

ls -l *${YEAR}*.txt | grep -v TALK 
