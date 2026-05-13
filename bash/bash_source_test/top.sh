#!/bin/bash

echo top.sh BS[0] ${BASH_SOURCE[0]}
echo top.sh BS[1] ${BASH_SOURCE[1]}
echo top.sh BS[2] ${BASH_SOURCE[2]}
echo top.sh BS[#] ${#BASH_SOURCE[@]}

source sub.sh


