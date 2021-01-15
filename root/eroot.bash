eroot-vi(){ vi ${BASH_SOURCE:-$(env-home)/root/root.bash} ; }
eroot-info(){ cat << EOI

   === $FUNCNAME 

    eroot-version-default : $(eroot-version-default)
    ROOT_VERSION         : $ROOT_VERSION  (set this in .bash_profile before "env-" to override default)
   eroot-version          : $(eroot-version $*)
    which root           : $(which root 2>/dev/null)

  Quantities derived from the eroot-version  :

    ROOTSYS      : $ROOTSYS   (exported into env by "eroot-" precursor ) 
   eroot-name     : $(eroot-name $*)
   eroot-nametag  : $(eroot-nametag $*)
   eroot-url      : $(eroot-url $*)
   eroot-rootsys  : $(eroot-rootsys $*)
   eroot-base     : $(eroot-base $*) 

   eroot-mode     : $(eroot-mode $*)
      if not "binary" source is assumed



     longterm default root_v5.21.04

    == C : rpath appears not to be enabled by default on Linux ? ==  

   After default ./configure  without any options/feature settings ... check what ./bin/eroot-config has to say 
{{{
[blyth@cms01 root]$ ./configure 
...
[blyth@cms01 root]$ ./bin/eroot-config --ldflags
-m32

[blyth@cms01 root]$ ./bin/eroot-config --cflags
-pthread -m32 -I/data/env/local/root/root_v5.26.00.source/root/./include

[blyth@cms01 root]$ ./bin/eroot-config --libs
-L/data/env/local/root/root_v5.26.00.source/root/./lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -pthread -lm -ldl -rdynamic

[blyth@cms01 root]$ ./bin/eroot-config --auxlibs
-pthread -lm -ldl -rdynamic
}}}


  Yep rpath is not enabled by default on Linux root_v5.26.00.source

{{{
    [blyth@cms01 root]$ ./configure --enable-rpath 
    ...
    [blyth@cms01 root]$ ./bin/eroot-config --auxlibs
    -pthread -Wl,-rpath,/data/env/local/root/root_v5.26.00.source/root/./lib -lm -ldl -rdynamic

    [blyth@cms01 root]$ ./bin/eroot-config --cflags
    -pthread -m32 -I/data/env/local/root/root_v5.26.00.source/root/./include

    [blyth@cms01 root]$ ./bin/eroot-config --libs
    -L/data/env/local/root/root_v5.26.00.source/root/./lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -pthread -Wl,-rpath,/data/env/local/root/root_v5.26.00.source/root/./lib -lm -ldl -rdynamic

}}}


   == getting Eve to load without LIBPATH ==

     eroot-libdeps RGL | sh
     eroot-libdeps Eve | sh




EOI
}



eroot-version-default(){
  local def="5.21.04"
  local jmy="5.22.00"   ## has eve X11 issues 
  local new="5.23.02" 
  local now="5.24.00" 
  local try="5.26.00e" 
  #local try="5.28.00" 
  case ${1:-$NODE_TAG} in 
     G) echo $try ;;
     C|C2|N) echo $try ;;
     *) echo $try ;;
  esac
}

eroot-archtag(){
   [ "$(eroot-mode)" != "binary" ] && echo "source" && return 0 
   case ${1:-$NODE_TAG} in 
      C) echo Linux-slc4-gcc3.4 ;;
      G) echo macosx-powerpc-gcc-4.0 ;;
      *) echo Linux-slc4-gcc3.4 ;;
   esac
}

#eroot-mode(){    echo binary  ; }
eroot-mode(){    echo -n  ; }
eroot-version(){ echo ${ROOT_VERSION:-$(eroot-version-default)} ; }
eroot-name(){    echo root_v$(eroot-version $*) ; }
eroot-nametag(){ echo $(eroot-name $*).$(eroot-archtag $*) ; }
eroot-rootsys(){ echo $(local-base $1)/root/$(eroot-nametag $1)/root ; }
eroot-base(){    echo $(dirname $(dirname $(eroot-rootsys $*))) ; }
eroot-url(){     echo ftp://root.cern.ch/root/$(eroot-nametag $*).tar.gz ; }


eroot-libdir(){  echo $(eroot-rootsys)/lib ; }


eroot-env(){
  elocal-
  export ROOT_NAME=$(eroot-name)
  export ROOTSYS=$(eroot-rootsys)
  eroot-path
  eroot-aliases
}

eroot-path(){
  [ ! -d $ROOTSYS ] && return 0
  env-prepend $ROOTSYS/bin
  env-llp-prepend $ROOTSYS/lib
  env-pp-prepend $ROOTSYS/lib
}

eroot-aliases(){
  #alias root="root -l"
  alias rh="tail -100 $HOME/.root_hist"
}

eroot-paths(){
  env-llp
  env-pp  
}

eroot-usage(){  eroot-info ; cat << EOU

  === $FUNCNAME 

  Check what you are getting :   
        ROOT_VERSION=5.24.00 eroot-info
        ROOT_VERSION=5.24.00 eroot-get

   eroot-           :  hook into these functions invoking eroot-env
   eroot-get        :  download and unpack
   eroot-configure  :     
   eroot-build      :

   eroot-path       :
         invoked by the precursor, sets up PATH (DY)LD_LIBRARY_PATH and PYTHONPATH

   eroot-pycheck
         http://root.cern.ch/root/HowtoPyROOT.html

   eroot-test-eve

   eroot-ps         : list root.exe processes
   eroot-killall    : kill root.exe processes

  eroot-c2py  

       cat alice_esd.C | eroot-c2py > alice_esd.py



EOU
}

eroot-ps(){      ps aux | grep root.exe ; }
eroot-killall(){ killall root.exe ; }
eroot-c2py(){    perl -p -e 's,\:\:,.,g' -  | perl -p -e 's,\->,.,g' - | perl -p -e 's,new ,,g' - | perl -p -e 's,;,,g' -  ; }
eroot-pycheck(){ python -c "import ROOT as _ ; print _.__file__ " ; }
eroot-signal(){ find $ROOTSYS -name '*.h' -exec grep -H SIGNAL {} \; ; }

eroot-find--(){
  local ext 
  for ext in $* ; do
      echo -n \ \-name \'*.$ext\' -o  
  done 
}
eroot-find-(){
   cat << EOC
find . -type f \( $(eroot-find-- cxx c hh py) -name '*.py'  \) -exec grep -H \$* {} \;
EOC
}
eroot-find(){
   cd $(eroot-rootsys)
   eval $(eroot-find- $*)
}


eroot-get(){
   local msg="=== $FUNCNAME :"
   local base=$(eroot-base)
   [ ! -d "$base" ] && mkdir -p $base 
   cd $base  ## 2 levels above ROOTSYS , the 
   local n=$(eroot-nametag)
   [ ! -f $n.tar.gz ] && curl -O $(eroot-url)
   [ ! -d $n/root   ] && mkdir $n && tar  -C $n -zxvf $n.tar.gz 
   ## unpacked tarballs create folder called "root"
}

eroot-build(){
   cd $(eroot-rootsys)
   export ROOTSYS=$(eroot-rootsys)
   echo ROOTSYS is $(eroot-rootsys)

   if [ "$(uname)" == "Linux" ]; then
      ./configure --enable-rpath
   else
      ./configure 
   fi 

   screen make
}

eroot-c(){ cd $(eroot-rootsys)/$1 ; }
eroot-cd(){ cd $(eroot-rootsys)/$1 ; }
eroot-eve(){ cd $(eroot-rootsys)/tutorials/eve ; }



eroot-test-tute(){
  local dir=$1
  local name=$2
  local msg="=== $FUNCNAME :"
  local iwd=$PWD

  cd $dir  
  [ ! -f "$name" ] && echo $msg no such script $PWD/$name  && cd $iwd && return 1
 
  eroot-config --version
 
  local cmd="root $name"
  echo $msg $cmd
  eval $cmd
}

eroot-test-eve(){ eroot-test-tute $ROOTSYS/tutorials/eve $* ; }
eroot-test-gl(){  eroot-test-tute $ROOTSYS/tutorials/gl  $* ; }



eroot-usage-deprecated(){ cat << EOX

    After changing the root version you will need to run :
        cmt-gensitereq

    This informs CMT of the change via re-generation of
     the non-managed :
         $ENV_HOME/externals/site/cmt/requirements
    containing the ROOT_prefix variable 

    This works via the ROOT_CMT envvar that is set by eroot-env, such as: 
       env | grep _CMT
       ROOT_CMT=ROOT_prefix:/data/env/local/root/root_v5.21.04.source/root

   Changing root version will require rebuilding libs that are 
   linked against the old root version, that includes dynamically created libs

EOX
}


eroot-libdiddle(){

   local nam=PyROOT
   local lib=lib$nam.so
   local tmp=/tmp/$USER/env/$FUNCNAME && mkdir -p $tmp && cd $tmp

   [ -f $lib ]   && rm $lib
   [ ! -f $lib ] && cp $(eroot-libdir)/$lib .
   
   echo $msg otool -D 
   otool -D $lib

   echo $msg otool -L 
   otool -L $lib

   #
   # attempt 1 :
   #     absolutize the libPyROOT.so install name
   #     and set dependents relative to that ... @loader_path/../libOther.so
   #        *  nope it appears that the @loader_path is not understood by root ?
   #
   # attempt 2 :
   #     just absolutize all dependent libs 
   #     and leave the lib itself at @rpath/libPyROOT.so
   #        * allows to load with 
   #             DYLD_LIBRARY_PATH=$(env-libdir) python -c "import ROOT ; ROOT.gSystem.Load('librootmq')  "
   #             DYLD_LIBRARY_PATH=$(env-libdir) ipython 
   #                   import ROOT
   #                   ROOT.gSystem.Load('librootmq')
   #                   ROOT.gMQ.Create()
   #                   ROOT.gMQ.SendString("hello")
   #
   #  ... if i could edit the rpath in a virtualenv python binary then could dispense
   #      with libpath setting at expense of picking the right "env" python, so comes down to PATH 
   #  ... only worth expense if having an "env" python binary has sufficient other benefits  
   #
   #    OR could create a /bin/sh wrapper called "python" that sets up the needed env then execs 
   #       the real python     
   #        ... low-tech way has advantage of being easily understood
   #            and not depending on uncommon os specific commands or diddling 
   #            ... not as expensive as diddling 
   # 
   #         * isolated site-packages ?
   #         * installation can pre-load the virtualized site-packages, with deps like ipython
   #           removing the need to cd ~/a/AbtViz or wherever
   #
   #

   #install_name_tool -id $(eroot-libdir)/libPyROOT.so $(eroot-libdir)/libPyROOT.so
   #install_name_tool -id @loader_path/../libPyROOT.so libPyROOT.so
   local cmd
   local deps="Core Cint RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread Reflex"
   for dep in $deps ; do
      #cmd="install_name_tool -change  @rpath/lib$dep.so @loader_path/../lib$dep.so libPyROOT.so"
      cmd="install_name_tool -change  @rpath/lib$dep.so $(eroot-libdir)/lib$dep.so libPyROOT.so"
      echo $cmd
      eval $cmd
   done
   echo after diddline 
   otool -L $lib

   cp libPyROOT.so $(eroot-libdir)/libPyROOT.so.diddled
}

eroot-first(){ echo $1 ; }
eroot-libdeps(){
   local nam=${1:-Eve}
   local lib=lib$nam.so
   local first
   local line
   otool -L $(eroot-libdir)/$lib | grep .so | while read line ; do
      first=$(eroot-first $line)
      case ${first:0:1} in
        /) echo -n ;;
        @) $FUNCNAME-xpath $lib $first ;;
      esac
   done
}
eroot-libdeps-xpath(){
   local parent=$1
   local child=$2

   local diff=$(( ${#child} - ${#parent} ))
   local end=${child:$diff}
   if [ "$end" == "$parent" ]; then
      return
   fi 

   #echo $FUNCNAME $parent ${#parent} $child ..${child:0:9}..  ${#child}  $diff  $end

   if [ "${child:0:6}" == "@rpath" ]; then 
       local cmd="install_name_tool -change $child $(eroot-libdir)/${child:7} $(eroot-libdir)/$parent"
       echo $cmd
   fi


}





eroot-libdiddle-place(){
   cd `eroot-libdir`
   mv libPyROOT.so libPyROOT.so.keep 
   mv libPyROOT.so.diddled libPyROOT.so 
}





eroot-testload(){
   local nam=${1:-rootmq} 
   cd
   eroot-testload-py    $nam $(env-libdir)
   eroot-testload-cint  $nam $(env-libdir)
}

eroot-testload-success(){ echo $FUNCNAME $* ; }
eroot-testload-fail(){    echo $FUNCNAME $* ; }

eroot-testload-env(){
   case $(uname) in
      Darwin) echo DYLD_LIBRARY_PATH=$1 ;;
           *) echo LD_LIBRARY_PATH=$1  ;;
   esac
}


eroot-testload-cint-(){ cat << EOM
{
    gSystem->Exit(gSystem->Load("lib$1"));
}
EOM
}
eroot-testload-cint(){
    local msg="=== $FUNCNAME :"
    local nam=${1:-rootmq}
    shift
    local tmp=/tmp/$USER/env/$FUNCNAME/$nam.C && mkdir -p $(dirname $tmp) 
    $FUNCNAME- $nam  > $tmp
    local path 
    local cmd
    for path in "" $* ; do
        cmd="$(eroot-testload-env $path) root -b -q $tmp"
        echo $msg $cmd
        eval $cmd > /dev/null 2>&1 && eroot-testload-success $FUNCNAME $nam $path || eroot-testload-fail $FUNCNAME $nam $path  
    done
}

eroot-testload-py(){
    local msg="=== $FUNCNAME :"
    local nam=${1:-rootmq}
    shift
    local path
    local cmd
    for path in "" $* ; do
        cmd="$(eroot-testload-env $path) python -c \"import ROOT ; ROOT.gSystem.Exit(ROOT.gSystem.Load('lib$nam'))\""
        echo $msg $cmd
        eval $cmd > /dev/null 2>&1  && eroot-testload-success $FUNCNAME $nam $path || eroot-testload-fail $FUNCNAME $nam $path
    done 
}



