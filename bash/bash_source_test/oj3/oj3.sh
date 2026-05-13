#!/bin/bash

cd $(dirname $(realpath $BASH_SOURCE))
source ../TEST.sh

echo oj3/oj3.sh BS    $BASH_SOURCE
echo oj3/oj3.sh BS[0] ${BASH_SOURCE[0]}
echo oj3/oj3.sh BS[1] ${BASH_SOURCE[1]}
echo oj3/oj3.sh BS[2] ${BASH_SOURCE[2]}
echo oj3.oj3.sh BS[#] ${#BASH_SOURCE[@]}



