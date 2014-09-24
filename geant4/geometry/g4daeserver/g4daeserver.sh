#!/bin/bash -l

g4daeserver-

g4daeserver-notes(){ cat << EON

TODO

Make these more similar 

EON
}


g4daeserver-vrun(){
  export-;
  export-export;
  $(g4daeserver-vdir)/bin/python $(g4daeserver-dir)/g4daeserver.py $*
}

g4daeserver-belle7(){
   python- source
   g4daeserver.py -w "127.0.0.1:8080 fcgi" 
}

g4daeserver-default(){
   g4daeserver.py $* 
}

case $NODE_TAG in 
  N) g4daeserver-belle7 $* ;;
  D) g4daeserver-vrun $* ;;
  *) g4daeserver-default $* ;;
esac



