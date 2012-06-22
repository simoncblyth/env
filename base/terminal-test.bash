#!/bin/bash

terminal-test-usage(){ cat << EOU

EOU
}

terminal-test(){
   local d=${1:-1}
   local r="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
   for i in $r
   do
      msg="$i sleeping $d " 
      test -t 0 && echo "$msg interactive" || echo "$msg not interactive"
      sleep $d
   done
}

terminal-test

