#!/bin/bash 

msg="=== $BASH_SOURCE :"
dir=$(dirname $BASH_SOURCE)
name=$(basename $BASH_SOURCE)
stem=${name/.sh}
applescript=$dir/$stem.applescript

echo $msg invoking $applescript from PWD $PWD
osascript $applescript $* 

