root-vi(){ vi ${BASH_SOURCE:-$(env-home)/root/root.bash} ; }
root-info(){ cat << EOI

   === $FUNCNAME 

    root-version-default : $(root-version-default)
    ROOT_VERSION         : $ROOT_VERSION  (set this in .bash_profile before "env-" to override default)
   root-version          : $(root-version $*)
    which root           : $(which root 2>/dev/null)

  Quantities derived from the root-version  :

    ROOTSYS      : $ROOTSYS   (exported into env by "root-" precursor ) 
   root-name     : $(root-name $*)
   root-nametag  : $(root-nametag $*)
   root-url      : $(root-url $*)
   root-rootsys  : $(root-rootsys $*)
   root-base     : $(root-base $*) 

   root-mode     : $(root-mode $*)
      if not "binary" source is assumed



     longterm default root_v5.21.04

    == C : rpath appears not to be enabled by default on Linux ? ==  

   After default ./configure  without any options/feature settings ... check what ./bin/root-config has to say 
{{{
[blyth@cms01 root]$ ./configure 
...
[blyth@cms01 root]$ ./bin/root-config --ldflags
-m32

[blyth@cms01 root]$ ./bin/root-config --cflags
-pthread -m32 -I/data/env/local/root/root_v5.26.00.source/root/./include

[blyth@cms01 root]$ ./bin/root-config --libs
-L/data/env/local/root/root_v5.26.00.source/root/./lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -pthread -lm -ldl -rdynamic

[blyth@cms01 root]$ ./bin/root-config --auxlibs
-pthread -lm -ldl -rdynamic
}}}


  Yep rpath is not enabled by default on Linux root_v5.26.00.source

{{{
    [blyth@cms01 root]$ ./configure --enable-rpath 
    ...
    [blyth@cms01 root]$ ./bin/root-config --auxlibs
    -pthread -Wl,-rpath,/data/env/local/root/root_v5.26.00.source/root/./lib -lm -ldl -rdynamic

    [blyth@cms01 root]$ ./bin/root-config --cflags
    -pthread -m32 -I/data/env/local/root/root_v5.26.00.source/root/./include

    [blyth@cms01 root]$ ./bin/root-config --libs
    -L/data/env/local/root/root_v5.26.00.source/root/./lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -pthread -Wl,-rpath,/data/env/local/root/root_v5.26.00.source/root/./lib -lm -ldl -rdynamic

}}}


   == getting Eve to load without LIBPATH ==

     root-libdeps RGL | sh
     root-libdeps Eve | sh




EOI
}



root-version-default(){
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

root-archtag(){
   [ "$(root-mode)" != "binary" ] && echo "source" && return 0 
   case ${1:-$NODE_TAG} in 
      C) echo Linux-slc4-gcc3.4 ;;
      G) echo macosx-powerpc-gcc-4.0 ;;
      *) echo Linux-slc4-gcc3.4 ;;
   esac
}

#root-mode(){    echo binary  ; }
root-mode(){    echo -n  ; }
root-version(){ echo ${ROOT_VERSION:-$(root-version-default)} ; }
root-name(){    echo root_v$(root-version $*) ; }
root-nametag(){ echo $(root-name $*).$(root-archtag $*) ; }
root-rootsys(){ echo $(local-base $1)/root/$(root-nametag $1)/root ; }
root-base(){    echo $(dirname $(dirname $(root-rootsys $*))) ; }
root-url(){     echo ftp://root.cern.ch/root/$(root-nametag $*).tar.gz ; }


root-libdir(){  echo $(root-rootsys)/lib ; }


root-env(){
  elocal-
  export ROOT_NAME=$(root-name)
  export ROOTSYS=$(root-rootsys)
  root-path
  root-aliases
}

root-path(){
  [ ! -d $ROOTSYS ] && return 0
  env-prepend $ROOTSYS/bin
  env-llp-prepend $ROOTSYS/lib
  env-pp-prepend $ROOTSYS/lib
}

root-aliases(){
  #alias root="root -l"
  alias rh="tail -100 $HOME/.root_hist"
}

root-paths(){
  env-llp
  env-pp  
}

root-usage(){  root-info ; cat << EOU

  === $FUNCNAME 

  Check what you are getting :   
        ROOT_VERSION=5.24.00 root-info
        ROOT_VERSION=5.24.00 root-get

   root-           :  hook into these functions invoking root-env
   root-get        :  download and unpack
   root-configure  :     
   root-build      :

   root-path       :
         invoked by the precursor, sets up PATH (DY)LD_LIBRARY_PATH and PYTHONPATH

   root-pycheck
         http://root.cern.ch/root/HowtoPyROOT.html

   root-test-eve

   root-ps         : list root.exe processes
   root-killall    : kill root.exe processes

  root-c2py  

       cat alice_esd.C | root-c2py > alice_esd.py



EOU
}

root-ps(){      ps aux | grep root.exe ; }
root-killall(){ killall root.exe ; }
root-c2py(){    perl -p -e 's,\:\:,.,g' -  | perl -p -e 's,\->,.,g' - | perl -p -e 's,new ,,g' - | perl -p -e 's,;,,g' -  ; }
root-pycheck(){ python -c "import ROOT as _ ; print _.__file__ " ; }
root-signal(){ find $ROOTSYS -name '*.h' -exec grep -H SIGNAL {} \; ; }

root-find--(){
  local ext 
  for ext in $* ; do
      echo -n \ \-name \'*.$ext\' -o  
  done 
}
root-find-(){
   cat << EOC
find . -type f \( $(root-find-- cxx c hh py) -name '*.py'  \) -exec grep -H \$* {} \;
EOC
}
root-find(){
   cd $(root-rootsys)
   eval $(root-find- $*)
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

root-build(){
   cd $(root-rootsys)
   export ROOTSYS=$(root-rootsys)
   echo ROOTSYS is $(root-rootsys)

   if [ "$(uname)" == "Linux" ]; then
      ./configure --enable-rpath
   else
      ./configure 
   fi 

   screen make
}

root-c(){ cd $(root-rootsys)/$1 ; }
root-cd(){ cd $(root-rootsys)/$1 ; }
root-eve(){ cd $(root-rootsys)/tutorials/eve ; }



root-test-tute(){
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



root-usage-deprecated(){ cat << EOX

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

EOX
}


root-libdiddle(){

   local nam=PyROOT
   local lib=lib$nam.so
   local tmp=/tmp/$USER/env/$FUNCNAME && mkdir -p $tmp && cd $tmp

   [ -f $lib ]   && rm $lib
   [ ! -f $lib ] && cp $(root-libdir)/$lib .
   
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

   #install_name_tool -id $(root-libdir)/libPyROOT.so $(root-libdir)/libPyROOT.so
   #install_name_tool -id @loader_path/../libPyROOT.so libPyROOT.so
   local cmd
   local deps="Core Cint RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread Reflex"
   for dep in $deps ; do
      #cmd="install_name_tool -change  @rpath/lib$dep.so @loader_path/../lib$dep.so libPyROOT.so"
      cmd="install_name_tool -change  @rpath/lib$dep.so $(root-libdir)/lib$dep.so libPyROOT.so"
      echo $cmd
      eval $cmd
   done
   echo after diddline 
   otool -L $lib

   cp libPyROOT.so $(root-libdir)/libPyROOT.so.diddled
}

root-first(){ echo $1 ; }
root-libdeps(){
   local nam=${1:-Eve}
   local lib=lib$nam.so
   local first
   local line
   otool -L $(root-libdir)/$lib | grep .so | while read line ; do
      first=$(root-first $line)
      case ${first:0:1} in
        /) echo -n ;;
        @) $FUNCNAME-xpath $lib $first ;;
      esac
   done
}
root-libdeps-xpath(){
   local parent=$1
   local child=$2

   local diff=$(( ${#child} - ${#parent} ))
   local end=${child:$diff}
   if [ "$end" == "$parent" ]; then
      return
   fi 

   #echo $FUNCNAME $parent ${#parent} $child ..${child:0:9}..  ${#child}  $diff  $end

   if [ "${child:0:6}" == "@rpath" ]; then 
       local cmd="install_name_tool -change $child $(root-libdir)/${child:7} $(root-libdir)/$parent"
       echo $cmd
   fi


}





root-libdiddle-place(){
   cd `root-libdir`
   mv libPyROOT.so libPyROOT.so.keep 
   mv libPyROOT.so.diddled libPyROOT.so 
}





root-testload(){
   local nam=${1:-rootmq} 
   cd
   root-testload-py    $nam $(env-libdir)
   root-testload-cint  $nam $(env-libdir)
}

root-testload-success(){ echo $FUNCNAME $* ; }
root-testload-fail(){    echo $FUNCNAME $* ; }

root-testload-env(){
   case $(uname) in
      Darwin) echo DYLD_LIBRARY_PATH=$1 ;;
           *) echo LD_LIBRARY_PATH=$1  ;;
   esac
}


root-testload-cint-(){ cat << EOM
{
    gSystem->Exit(gSystem->Load("lib$1"));
}
EOM
}
root-testload-cint(){
    local msg="=== $FUNCNAME :"
    local nam=${1:-rootmq}
    shift
    local tmp=/tmp/$USER/env/$FUNCNAME/$nam.C && mkdir -p $(dirname $tmp) 
    $FUNCNAME- $nam  > $tmp
    local path 
    local cmd
    for path in "" $* ; do
        cmd="$(root-testload-env $path) root -b -q $tmp"
        echo $msg $cmd
        eval $cmd > /dev/null 2>&1 && root-testload-success $FUNCNAME $nam $path || root-testload-fail $FUNCNAME $nam $path  
    done
}

root-testload-py(){
    local msg="=== $FUNCNAME :"
    local nam=${1:-rootmq}
    shift
    local path
    local cmd
    for path in "" $* ; do
        cmd="$(root-testload-env $path) python -c \"import ROOT ; ROOT.gSystem.Exit(ROOT.gSystem.Load('lib$nam'))\""
        echo $msg $cmd
        eval $cmd > /dev/null 2>&1  && root-testload-success $FUNCNAME $nam $path || root-testload-fail $FUNCNAME $nam $path
    done 
}



