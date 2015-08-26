#!/bin/bash -l

g4daeserver-

g4daeserver-notes(){ cat << EON

TODO

Make these more similar by vpython adoption on N 

EON
}

g4daeserver-runenv(){
  export-;
  export-export;
}

g4daeserver-vrun(){ $(g4daeserver-vdir)/bin/python $(g4daeserver-dir)/g4daeserver.py $* ; }

g4daeserver-belle7(){
  python- source
  g4daeserver.py --webpy fcgi 
}

g4daeserver-default(){
  g4daeserver.py $* 
}

g4daeserver-main(){
  g4daeserver-runenv
  case $NODE_TAG in 
    N) g4daeserver-belle7 $* ;;
    D) g4daeserver-vrun $* ;;
    *) g4daeserver-default $* ;;
  esac
}

g4daeserver-main $*


