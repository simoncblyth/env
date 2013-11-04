#!/bin/bash -l
env-

daeserver-belle7(){
   python- source
   daeserver.py -w "127.0.0.1:8080 fcgi" 
}

daeserver-default(){
    daeserver.py 
}

case $NODE_TAG in 
  N) daeserver-belle7 ;;
  *) daeserver-default ;;
esac



