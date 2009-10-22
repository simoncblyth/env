vip-src(){      echo vip/vip.bash ; }
vip-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vip-src)} ; }
vip-vi(){       vi $(vip-source) ; }

vip-base(){ echo $(local-base)/env/vip ; }
#vip-name(){ echo ${1:-$VIP_NAME} ; }
vip-name(){ basename ${VIRTUAL_ENV:-$1} ; } 
vip-dir(){  echo $(vip-base)/$(vip-name $1) ; }
vip-srcdir(){  echo $(vip-base)/src ; }
vip-cd(){   cd $(vip-dir $*); }
vip-mate(){ mate $(vip-dir $*) ; }
vip-activate(){    . $(vip-dir $*)/bin/activate ;  }
vip-reqpath(){ echo $(vip-dir $*)/requirememts.txt ; }
vip-deactivate(){  deactivate ; }

vip-env(){ echo -n ; }
vip--(){ 
   local msg="=== $FUNCNAME :"
   local cmd="pip -E $(vip-dir) install --src=$(vip-srcdir) --no-deps $*"
   echo $msg \"$cmd\"
   eval $cmd
}

vip-usage(){
  cat << EOU
     vip-src : $(vip-src)
     vip-dir : $(vip-dir)

     vip-install <name>

        Usage : 
           cat production.pip | vip-install dbi
  
          Background, pip options : 
            -s  :  use site packages when creating v : eg for MySQLdb that needs to come from system
                   (DOES NOT WORK ... SO CREATE virtualenv SEPARATELY)

      --no-deps :  so the requirements file is fully in control of versions 
       -E \$dir :  v directory to use OR create
       -r \$pip :  requirements file specifying the versions to use
    --src \$src : directory to check out/clone editables into, avoid repeating slow clones      

      the srcdir is located one step above the individual env allowing editable paths  :
            -e ../src/


      TODO:

         


EOU
}


vip-install(){
  local msg="=== $FUNCNAME :"
  local nam=$(vip-name $*)
  shift
  local cmd
  local sdir=$(vip-srcdir)
  [ ! -d "$sdir" ] && mkdir -p $sdir
  local dir=$(vip-dir $nam)
  local req=$(vip-reqpath $nam)   
  [ ! -d "$dir" ] && mkdir -p $dir

  cmd="cd $dir"
  echo $msg \"$cmd\"
  eval $cmd

  cmd="virtualenv $dir "
  echo $msg \"$cmd\"
  eval $cmd

  echo $msg stream stdin into the requirements $req
  cat - > $req

  cmd="pip -E $dir install --no-deps -r $req --src=$sdir --log=$dir/pip.log   $* "
  echo $msg installation into $dir ... based on the requirements : $req
  echo $msg \"$cmd\"  
  eval $cmd
}




vip-freeze(){
  local msg="=== $FUNCNAME :"
  local msg="=== $FUNCNAME :"
  local nam=$(vip-name $*)
  local dir=$(vip-dir $nam)
  shift

  local req=$(vip-reqpath $nam)   
  local tmp=/tmp/env/$FUNCNAME/$(basename $pip) && mkdir -p $(dirname $tmp)
  local cmd
  if [ -f "$req" ] ; then
     cmd="pip -E $dir freeze -r $req "   ## -r 
  else
     cmd="pip -E $dir freeze "
  fi
  echo $msg \"$cmd\"
  echo $msg freezing the state of python into $tmp ... for possible updating of $req
  eval $cmd > $tmp

  if [ -f "$req" ]; then 
     diff $req $tmp
     echo $msg NOT COPYING AS TOO MESSY FOR AUTOMATION ... DO THAT YOURSELF WITH : \"cp $tmp $req\"
  else
     echo $msg copying initial pip freeze to $req
     cp $tmp $req
  fi
}






vip-build(){
   vip-get 
   [ ! $? -eq 0 ]  && return 1
   env-build
}

vip-wipe(){
   local msg="=== $FUNCNAME :"
   local cmd="rm -rf $(vip-dir) " 
   local ans
   read -p "$msg proceed with \"$cmd\"  enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipped && return 0
   eval $cmd
}

vip-get-deprecated(){
   local dir=$(dirname $(vip-dir)) &&  mkdir -p $dir && cd $dir
   local msg="=== $FUNCNAME :"
   [ "$(which virtualenv)" == "" ] && echo $msg missing virtualenv && return 1
   [ "$(which pip)" == "" ] && echo $msg missing pip && return 1
   [ "$(which hg)" == "" ] && echo $msg missing hg && return 1
   [ ! -d "$(vip-dir)" ] && virtualenv $(vip-dir) || echo $msg virtualenv dir $(vip-dir) exists already skipping virtualenv creation 
   ! vip-activate && echo "$msg failed to activate " && return 1 
   [ "$(which python)" != "$(vip-dir)/bin/python" ] && echo $msg ABORT failed to setup virtual python $(which python)   && return 1
   python -c "import MySQLdb" 2> /dev/null
   [ ! $? -eq 0 ] && echo $msg ABORT missing MySQLdb ... see pymysql-build or system install if using system python  && return 1
   return 0
}

## handle such node specifics in .bash_profile 
vip-env-C(){
   ## system python on cms01 : 2.3.4 is too old to use ... so the basis of the virtualenv is the source python
   python- source 
}

vip-env-deprecated(){      
   elocal- ; 
   case $NODE_TAG in  
     C) vip-env-C ;;
   esac
  
   #vip-activate ;  
   ## this distinguishes deployed running and debug running 
   #case $(vip-mode) in 
   #  dev) export ENV_PRIVATE_PATH=$HOME/.bash_private ;; 
   #    *) export ENV_PRIVATE_PATH=$(apache-private-path) ;;
   #esac
}
