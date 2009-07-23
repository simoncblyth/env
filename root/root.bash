root-vi(){ vi $BASH_SOURCE ; }
root-info(){

  cat << EOI

   root-mode     : $(root-mode $*)
      if not "binary" source is assumed

   root-version  : $(root-version $*)
   root-name     : $(root-name $*)
   root-nametag  : $(root-nametag $*)
   root-url      : $(root-url $*)
   root-rootsys  : $(root-rootsys $*)
   root-base     : $(root-base $*) 
 

    ROOTSYS    : $ROOTSYS
    which root : $(which root)

    After changing the root version you will need to run :
        cmt-gensitereq

    This informs CMT of the change via re-generation of
     the non-managed :
         $ENV_HOME/externals/site/cmt/requirements
    containing the ROOT_prefix variable 

    This works via the ROOT_CMT envvar that is set by root-env, such as: 
       env | grep _CMT
       ROOT_CMT=ROOT_prefix:/data/env/local/root/root_v5.21.04.source/root

   Changing root version will require rebuilding libs that are 
   linked against the old root version, that includes dynamically created libs


   root-           :  hook into these functions invoking root-env
   root-get        :  download and unpack
   root-configure  :     
   root-build      :


   root-path       :
         invoked by the precursor, sets up PATH (DY)LD_LIBRARY_PATH and PYTHONPATH

   root-pycheck
         http://root.cern.ch/root/HowtoPyROOT.html

   root-evetest

   root-ps         : list root.exe processes
   root-killall    : kill root.exe processes



EOI

}

root-usage(){
   root-info
}


root-ps(){
  ps aux | grep root.exe
}

root-killall(){
   killall root.exe
}

root-archtag(){

   [ "$(root-mode)" != "binary" ] && echo "source" && return 0 

   case ${1:-$NODE_TAG} in 
      C) echo Linux-slc4-gcc3.4 ;;
      G) echo macosx-powerpc-gcc-4.0 ;;
      *) echo Linux-slc4-gcc3.4 ;;
   esac
}

root-nametag(){ 
   echo $(root-name $*).$(root-archtag $*) 
}


root-get(){

   local msg="=== $FUNCNAME :"
   
   local base=$(root-base)
   [ ! -d "$base" ] && mkdir -p $base 
   cd $base  ## 2 levels above ROOTSYS , the 
   local n=$(root-nametag)
   [ ! -f $n.tar.gz ] && curl -O $(root-url)
   [ ! -d $n/root   ] && mkdir $n && tar  -C $n -zxvf $n.tar.gz 
 
   ## unpacked tarballs create folder called "root"
}

root-version(){
  local def="5.21.04"
  local jmy="5.22.00"   ## has eve X11 issues 
  local new="5.23.02" 
  case ${1:-$NODE_TAG} in 
     C) echo $def ;;
     *) echo $def ;;
  esac
}

#root-mode(){    echo binary  ; }
root-mode(){    echo -n  ; }
root-name(){    echo root_v$(root-version $*) ; }
root-base(){    echo $(dirname $(dirname $(root-rootsys $*))) ; }
root-rootsys(){ echo $(local-base $1)/root/$(root-nametag $1)/root ; }
root-url(){     echo ftp://root.cern.ch/root/$(root-nametag $*).tar.gz ; }

root-cd(){ cd $(root-rootsys)/tutorials/eve ; }



root-env(){

  elocal-

  alias root="root -l"
  alias rh="tail -100 $HOME/.root_hist"
 
  export ROOT_NAME=$(root-name)
  export ROOTSYS=$(root-rootsys)
  
  ## pre-nuwa ... to be dropped      	
  export ROOT_CMT="ROOT_prefix:$ROOTSYS"

  root-path
}



root-path(){

  [ ! -d $ROOTSYS ] && return 0
  
  env-prepend $ROOTSYS/bin
  env-llp-prepend $ROOTSYS/lib
  env-pp-prepend $ROOTSYS/lib
}


root-paths(){
  env-llp
  env-pp  
}








root-pycheck(){
  python -c "import ROOT as _ ; print _.__file__ "

}


root-tute-test(){
  local dir=$1
  local name=$2
  local msg="=== $FUNCNAME :"
  local iwd=$PWD

  cd $dir  
  [ ! -f "$name" ] && echo $msg no such script $PWD/$name  && cd $iwd && return 1
 
  root-config --version
 
  local cmd="root $name"
  echo $msg $cmd
  eval $cmd
}

root-test-eve(){ root-test-tute $ROOTSYS/tutorials/eve $* ; }
root-test-gl(){  root-test-tute $ROOTSYS/tutorials/gl  $* ; }




