#!/usr/bin/env bash

usage(){ cat << EOU


~/e/tools/curl_check/curl_check.sh
CHECK=curl_check_0 ~/e/tools/curl_check/curl_check.sh
CHECK=curl_check_1 ~/e/tools/curl_check/curl_check.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))


#check=0
#check=1
check=2

export CHECK=${CHECK:-$check}

name=curl_check_$CHECK
bin=/tmp/$USER/env/tools/$name
mkdir -p $(dirname $bin)


vv="BASH_SOURCE PWD check CHECK name bin" 
for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done


gcc $name.cc -o $bin \
    -Wall -std=c++17 -lstdc++ \
    $(curl-config --cflags) \
    $(curl-config --libs) 
[ $? -ne 0 ] && echo $BASH_SOURCE - build error && exit 1 


$bin
[ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 2

exit 0


