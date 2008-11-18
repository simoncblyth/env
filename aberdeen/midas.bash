
midas-usage(){

   cat << EOU
   
       midas-name   : $(midas-name)
       midas-folder : $(midas-folder)


   
EOU




}



midas-env(){
   
  local- 
   
   
   
  export MIDAS_NAME=$(midas-name) 
  export MIDAS_FOLDER=$(midas-folder)
  export MIDAS_UNAME=$(midas-uname) 
  export MIDAS_USER=$(midas-user)

  # specifies where the exptab file is  
  #   documation incorrectly? implies that this points to the directory rather than the file
  # 
  export MIDAS_EXPTAB=$(midas-exptab)

  # env required for installation 
  #     http://midas.psi.ch/htmldoc/quickstart.html
  #     NB MIDAS_DIR is a used variable ...  see 
  #          http://midas.psi.ch/htmldoc/AppendixD.html#Environment_variables
  #
  export MIDASSYS=$(midas-folder)


}


midas-exptab(){
   echo $(midas-folder)/examples/experiment/exptab
}

midas-info(){
   env | grep MIDAS_
   
   local xtab=$(midas-exptab)
   if [ -f "$xtab" ]; then
     cat $xtab
  else
     echo $msg ERROR ..... no MIDAS_EXPTAB file $xtab 
  fi

}


midas-uname(){
   uname | tr "A-Z" "a-z"
}

midas-user(){
  case $NODE_TAG in
     C) echo dayabaysoft ;;
     *) echo $USER ;;
  esac     
}

midas-name(){
   case $NODE_TAG in
     *) echo midas-2.0.0 ;;
   esac  
}


midas-folder(){
   case $NODE_TAG in 
 G|C|P) echo $LOCAL_BASE/env/midas/$(midas-name) ;;
     *) echo $LOCAL_BASE/midas/$(midas-name) ;; 
   esac
}




midas-path(){

  local msg="=== $FUNCNAME :" 
  # only append the PATH if the path being appended is not there already 
  local mbin=$(midas-folder)/$(midas-uname)/bin
  [ ! -d "$mbin" ] && echo $msg no dir $mbin && return 0
  
  test $PATH == ${PATH/$mbin/} && PATH=$PATH:$mbin

  local xbin=$(midas-folder)/examples/experiment 
  test $PATH == ${PATH/$xbin/} && PATH=$PATH:$xbin
  
  echo $PATH | tr ":" "\n"
  
}


midas-get(){

  local dir=$(midas-folder)
  [ ! -d "$dir" ] && $SUDO mkdir -p $dir && $SUDO chown $USER $dir
  
  local tgz=$(midas-name).tar.gz
  #local url=http://midas.psi.ch/download/tar/$tgz
  local url=https://midas.psi.ch/download/tar/$tgz


  cd $dir

  [ -f $tgz          ] || curl -o $tgz $url
  [ -d $(midas-name) ] || tar zxvf $tgz

}


midas-make(){
  
  
  cd $(midas-folder)
  make
  make examples

#
# [g4pb:~] blyth$ midas-make
# cc -g -O2 -Wall -Wuninitialized -Iinclude -Idrivers -I../mxml -Ldarwin/lib -DINCLUDE_FTPLIB   -DHAVE_ROOT -D_REENTRANT -Wno-long-double -I/usr/local/root/root_v5.14.00b/root/include -DOS_LINUX -DOS_DARWIN -DHAVE_STRLCPY -fPIC -Wno-unused-function -o darwin/bin/consume examples/lowlevel/consume.c darwin/lib/libmidas.a -lpthread
# cc -g -O2 -Wall -Wuninitialized -Iinclude -Idrivers -I../mxml -Ldarwin/lib -DINCLUDE_FTPLIB   -DHAVE_ROOT -D_REENTRANT -Wno-long-double -I/usr/local/root/root_v5.14.00b/root/include -DOS_LINUX -DOS_DARWIN -DHAVE_STRLCPY -fPIC -Wno-unused-function -o darwin/bin/produce examples/lowlevel/produce.c darwin/lib/libmidas.a -lpthread
# cc -g -O2 -Wall -Wuninitialized -Iinclude -Idrivers -I../mxml -Ldarwin/lib -DINCLUDE_FTPLIB   -DHAVE_ROOT -D_REENTRANT -Wno-long-double -I/usr/local/root/root_v5.14.00b/root/include -DOS_LINUX -DOS_DARWIN -DHAVE_STRLCPY -fPIC -Wno-unused-function -o darwin/bin/rpc_test examples/lowlevel/rpc_test.c darwin/lib/libmidas.a -lpthread
# cc -g -O2 -Wall -Wuninitialized -Iinclude -Idrivers -I../mxml -Ldarwin/lib -DINCLUDE_FTPLIB   -DHAVE_ROOT -D_REENTRANT -Wno-long-double -I/usr/local/root/root_v5.14.00b/root/include -DOS_LINUX -DOS_DARWIN -DHAVE_STRLCPY -fPIC -Wno-unused-function -o darwin/bin/msgdump examples/basic/msgdump.c darwin/lib/libmidas.a -lpthread
# examples/basic/msgdump.c: In function 'main':
# examples/basic/msgdump.c:78: warning: format '%ld' expects type 'long int', but argument 2 has type 'DWORD'
# examples/basic/msgdump.c:118: warning: format '%ld' expects type 'long int', but argument 3 has type 'DWORD'
# cc -g -O2 -Wall -Wuninitialized -Iinclude -Idrivers -I../mxml -Ldarwin/lib -DINCLUDE_FTPLIB   -DHAVE_ROOT -D_REENTRANT -Wno-long-double -I/usr/local/root/root_v5.14.00b/root/include -DOS_LINUX -DOS_DARWIN -DHAVE_STRLCPY -fPIC -Wno-unused-function -o darwin/bin/minife examples/basic/minife.c darwin/lib/libmidas.a -lpthread
# examples/basic/minife.c:57: warning: return type defaults to 'int'
# examples/basic/minife.c: In function 'main':
# examples/basic/minife.c:78: error: too few arguments to function 'cm_register_transition'
# examples/basic/minife.c:79: error: too few arguments to function 'cm_register_transition'
# examples/basic/minife.c:88: error: 'EVENT_BUFFER_SIZE' undeclared (first use in this function)
# examples/basic/minife.c:88: error: (Each undeclared identifier is reported only once
# examples/basic/minife.c:88: error: for each function it appears in.)
# make: *** [darwin/bin/minife] Error 1
#
#       solved compilation issues by inserting "random" parameters ...
#   
#


}

midas-expt-make(){

 
  cd $(midas-folder)/examples/experiment
  
  make 

#
# g++ -g -I/usr/local/midas/midas-2.0.0/include -I/usr/local/midas/midas-2.0.0/drivers/camac -o analyzer /usr/local/midas/midas-2.0.0/darwin/lib/rmana.o analyzer.o adccalib.o adcsum.o scaler.o \
# /usr/local/midas/midas-2.0.0/darwin/lib/libmidas.a  -L/usr/local/root/root_v5.14.00b/root/lib -lCore -lCint -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lfreetype -lpthread -lm -ldl -Wl,-rpath,/usr/local/root/root_v5.14.00b/root/lib -lThread -lpthread
# /usr/bin/ld: unknown flag: -rpath
# collect2: ld returned 1 exit status
#
#     ... removed the -Wl,rpath,blah  in order to get the analyzer to build on Darwin
#      will probably need to set DYLD_LIBRARY_PATH to see the root libs instead
#
#   -rpath dir
#           Add  a  directory to the runtime library search path.  This is used
#           when linking an ELF executable with  shared  objects.   All  -rpath
#           arguments  are concatenated and passed to the runtime linker, which
#           uses them to locate shared objects at runtime.  The  -rpath  option
#           is  also  used  when  locating  shared  objects which are needed by
#           shared objects explicitly included in the link; see the description
#           of  the  -rpath-link option.  If -rpath is not used when linking an
#           ELF  executable,  the  contents   of   the   environment   variable
#           "LD_RUN_PATH" will be used if it is defined.
#
#           The -rpath option may also be used on SunOS.  By default, on SunOS,
#           the linker will form a runtime search  patch  out  of  all  the  -L
#           options  it  is  given.   If  a  -rpath option is used, the runtime
#           search path will be formed exclusively using  the  -rpath  options,
#           ignoring  the -L options.  This can be useful when using gcc, which
#           adds many -L options which may be on NFS mounted filesystems.
#
#           For compatibility with other ELF linkers, if the -R option is  fol-
#           lowed  by  a directory name, rather than a file name, it is treated
#           as the -rpath option.
#
#
#  
#
#



}


midas-expt-run(){


   cd $(midas-folder)/examples/experiment
   
   ./frontend
   
# [g4pb:/usr/local/midas/midas-2.0.0/examples/experiment] blyth$ ./frontend
# Frontend name          :     Sample Frontend
# Event buffer size      :     100000
# Buffer allocation      : 2 x 100000
# System max event size  :     4194304
# User max event size    :     10000
# User max frag. size    :     5242880
# # of events per buffer :     10
#
# Connect to experiment...
# OK
# Init hardware...
#    
#       fails with 
#           "Cannot open event buffer "SYSTEM" size 8388608, bm_open_buffer() status 218   "
#
#
#
#  ./analyzer
#  /Users/blyth/.rootlogon.C loading (created by root-use-rootlogon, invoked by root-use-conf, see env:trunk/dyw/root_use.bash ) 
#  Connect to experiment ...
#  Error: Experiment "" not defined.
#

   
}


midas-expt(){
  cd $(midas-folder)/examples/experiment
}


 
  



midas-expt-config(){
  
   
   cd $(midas-folder)/examples/experiment
   mkdir -p expts/test
    mkdir -p expts/test2
     mkdir -p expts/test3
   
   
   
   cat << EOC > exptab
# exptab file generated by midas-expt-config in midas.bash
#
#   exptname/exptdir/username
test $(midas-folder)/examples/experiment/expts/test $(midas-user)
test2 $(midas-folder)/examples/experiment/expts/test2 $(midas-user)
test3 $(midas-folder)/examples/experiment/expts/test3 $(midas-user)


EOC

  cat exptab


}



midas-install(){

  
  cd $(midas-folder)
  $SUDO make PREFIX=$(midas-folder) install

#
# to get the install to work on darwin removed the "-D" option from "install" an added 
#  several  "mkdir -p"  
#  and a   mkdir -p `dirname $(PREFIX)/$$file` ;  for the drivers 
#


}