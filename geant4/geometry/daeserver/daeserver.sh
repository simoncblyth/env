#!/bin/bash -l

daeserver-

daeserver-notes(){ cat << EON

TODO

Make these more similar 

EON
}


daeserver-vrun(){
  export-;
  export-export;
  $(daeserver-vdir)/bin/python $(daeserver-dir)/daeserver.py $*
}

daeserver-belle7(){
   python- source
   daeserver.py -w "127.0.0.1:8080 fcgi" 
}

daeserver-default(){
    daeserver.py $* 
}

case $NODE_TAG in 
  N) daeserver-belle7 $* ;;
  D) daeserver-vrun $* ;;
  *) daeserver-default $* ;;
esac



