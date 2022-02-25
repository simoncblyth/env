#!/bin/bash

# https://www.golinuxcloud.com/bash-split-string-into-array-linux/

myvar="string1,string2,string3"

# Here comma is our delimiter value
IFS="," read -ra myarray <<< "$myvar"

# quotes on the variable is due to a bug fixed in bash 4.3  
# according to 
# https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash


echo "My array: ${myarray[@]}"
echo "Number of elements in the array: ${#myarray[@]}"
