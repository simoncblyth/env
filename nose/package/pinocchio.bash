# === func-gen- : nose/package/pinocchio fgp nose/package/pinocchio.bash fgn pinocchio fgh nose/package
pinocchio-src(){      echo nose/package/pinocchio.bash ; }
pinocchio-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pinocchio-src)} ; }
pinocchio-vi(){       vi $(pinocchio-source) ; }
pinocchio-env(){      elocal- ; }
pinocchio-usage(){
  cat << EOU
     pinocchio-src : $(pinocchio-src)
     pinocchio-dir : $(pinocchio-dir)

    The stopwatch nosetests plugin enables the timing of test operation 
    and selection of faster tests based on the running time of prior test runs.

    Comment the figleaf part of the setup.py to avoid a runtime warning regarding
    missing figleaf dependency


EOU
}
pinocchio-dir(){ echo $(local-base)/env/nose/$(pinocchio-name) ; }
pinocchio-cd(){  cd $(pinocchio-dir); }
pinocchio-mate(){ mate $(pinocchio-dir) ; }
pinocchio-name(){ echo pinocchio-latest ; }
pinocchio-url(){ echo http://darcs.idyll.org/~t/projects/$(pinocchio-name).tar.gz ; }

pinocchio-get(){
   local dir=$(dirname $(pinocchio-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -f "$(pinocchio-name).tar.gz" ] && curl -O $(pinocchio-url) 
   [ ! -d "$(pinocchio-name)" ]  && tar zxvf $(pinocchio-name).tar.gz
}

pinocchio-build(){
   pinocchio-cd
   sudo easy_install .
   nosetests -p
}

pinocchio-times-(){ python -c "from cPickle import load ; d = load(open('.nose-stopwatch-times')) ; print '\n'.join(['%-100s:%s'% (k,d[k]) for k in sorted(d,key=$1)]) " ; }
pinocchio-bytime(){  pinocchio-times- 'lambda x:d[x]' ; }
pinocchio-byname(){  pinocchio-times- 'lambda x:x' ; }
pinocchio-test(){
   pinocchio-cd
   nosetests --with-stopwatch -v examples/test_stopwatch.py 
   [ ! -f .nose-stopwatch-times ] && echo $msg ERROR failed to create time record && return 1
   pinocchio-byname
}


