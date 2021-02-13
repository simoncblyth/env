#!/bin/bash 
abs=$(ls -1 *abstract.tex)
echo === $0 : check word count of abs $abs  
cat $abs | wc -w




