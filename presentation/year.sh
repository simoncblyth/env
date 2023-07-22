#!/bin/bash -l 

year=2023
YEAR=${YEAR:-$year}

ls -l *${YEAR}*.txt | grep -v TALK 
