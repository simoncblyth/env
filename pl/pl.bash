# === func-gen- : offline/pl/pl.bash fgp offline/pl/pl.bash fgn pl
pl-src(){      echo offline/pl/pl.bash ; }
pl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pl-src)} ; }
pl-dir(){      echo $(env-home $*)/$(dirname $(pl-src)) ; }
pl-vi(){       vi $(pl-source) ; }
pl-env(){      elocal- ; }


pl-projname(){ echo ${PL_PROJNAME:-helloworld} ; }
pl-confname(){ echo ${PL_CONFNAME:-development} ; }
pl-projdir(){  echo ${PL_PROJDIR:-$(pl-dir)/$(pl-projname)} ; }

pl-confpath(){  echo $(pl-projdir)/$(pl-confname).ini ; }


pl-usage(){
  cat << EOU

    Basis Coordinates :

        pl-projname : $(pl-projname)
        pl-projdir  : $(pl-projdir)
        pl-confname : $(pl-confname)

    Derived :

        pl-confpath : $(pl-confpath)


     pl-src : $(pl-src)

     pl-install
           set up virtual env and populate with Pylons

     pl-quickstart 
           set up project 

     pl-deploy 


EOU
}


pl-preq-install-yum(){  [ "$(which hg)" == "" ] && sudo yum  install mercurial ; } 
pl-preq-install-port(){ [ "$(which hg)" == "" ] && sudo port install mercurial ; }
pl-preq-install(){
   pkgr-
   case $(pkgr-cmd) in 
      yum) $FUNCNAME-yum ;;
     port) $FUNCNAME-port ;;
   esac
}

pl-preq(){
    local msg="=== $FUNCNAME :"
    echo $msg preqs for the baseline python ... not the virtualized one
    python-
    [ "$(python-version)"     != "2.4.3" ]  && echo $msg untested python version

    virtualenv-
    virtualenv-get
    [ "$(virtualenv-version)" != "1.3.3" ] && echo $msg untested virtualenv  
}

pl-srcfold(){  echo $(local-base $*)/env ; }
pl-srcnam(){   echo pldev ; }  
pl-srcdir(){   echo $(pl-srcfold $*)/$(pl-srcnam)/pylons ; }
pl-mate(){     mate $(pl-srcdir) ; }

pl-cd(){       cd $(pl-projdir) ; }   


pl-build(){

  local msg="=== $FUNCNAME :"
  [ -z "$VIRTUAL_ENV" ] && echo $msg ABORT are not inside virtualenv && return 1 
  [ "$(which python)" != "$VIRTUAL_ENV/bin/python" ] && echo  $msg ABORT wrong python && return 1

  pl-get
  pl-install
  pl-selinux 
  pl-eggcache
}




pl-get(){
   local msg="=== $FUNCNAME :"
   [ "$(which hg)" == "" ] && echo $msg no hg && return 1
   local dir=$(dirname $(pl-srcdir))
   local nam=$(basename $(pl-srcdir))

   mkdir -p $dir && cd $dir
   local cmd="hg clone http://bitbucket.org/bbangert/pylons/ $nam"
   echo $msg \"$cmd\" from $PWD
   eval $cmd 

   #hg clone https://www.knowledgetap.com/hg/pylons-dev Pylons
}

pl-install(){
   local msg="=== $FUNCNAME :"

   cd $(pl-srcdir)
   local cmd="python setup.py develop"
   echo $msg \"$cmd\"  ... from $PWD with $(which python)
   eval $cmd
}


pl-proj-deps(){
  pl-activate

  ## could be handled by adding requirements to the setup.py of the proj 
  easy_install configobj
  easy_install ipython  
  easy_install MySQL-python
}

  
pl-create(){
  pl-activate
  cd $(pl-dir)  
  local proj=${1:-$(pl-projname)}
  [ -d "$proj" ] && echo $msg ERROR proj $proj exists already && return 1 

  paster create -t pylons $proj 
  cd $proj

  ## this will get the dependencies, such as SQLAlchemy 
  python setup.py develop

  ## edit the development.ini adding DB coordinates etc..
  pl-conf

  # python-ln $(env-home) env   ## for env.base.private.Private access
}


pl-ini(){ 
   ini-
   INI_TRIPLET_DELIM="|" INI_FLAVOR="ini" ini-triplet-edit $(pl-confpath) $*  
}

pl-serve-(){
  case $(pl-confname) in
     development) echo paster serve --reload $(pl-confpath) ;;
               *) echo paster serve          $(pl-confpath) ;;
  esac
}

pl-serve(){
  local msg="=== $FUNCNAME :"
  local cmd=$($FUNCNAME-)
  echo $msg \"$cmd\"
  eval $cmd 
}



