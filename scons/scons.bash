# === func-gen- : scons/scons fgp scons/scons.bash fgn scons fgh scons
scons-src(){      echo scons/scons.bash ; }
scons-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scons-src)} ; }
scons-vi(){       vi $(scons-source) ; }
scons-env(){      elocal- ; }
scons-usage(){
  cat << EOU
     scons-src : $(scons-src)
     scons-dir : $(scons-dir)

   == installs ==

     port 
          sudo port install scons
     
            * installed v1.2.0.r3842 onto G, mostly into /opt/local/lib/scons-1.2.0
            * after move to py26 on G, installed scons 2.0.1 mostly into /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/scons-2.0.1/

     yum  installed v1.2.0.r3842  onto C,C2,N
         sudo yum install scons
         
     ipkg installed v1.2.0.r3842 
         sudo ipkg install scons

   == installation ==

     scons-dir-export
          defines SCONS_DIR envvar
          ... not needed by scons itself, but used by SCT to find SCons

     scons-dir-pth
          writes a python .pth file into python-site, which 
          enables SCT to find SCons without having to use an envvar


   == Issues ==

        simon:e blyth$ scons -c
        scons: *** No SConstruct file found.
        File "/opt/local/lib/scons-1.2.0/SCons/Script/Main.py", line 826, in _main

            you need to use : "sct -c" 


   == python setup issue (needs to be done once for each python) ==

       Traceback (most recent call last):
       File "/usr/local/env/scons/sct/wrapper.py", line 44, in <module>
           import SCons.Script
       ImportError: No module named SCons.Script


       use scons-dir-pth  



   == Questions ==
 
     1) why does "scons -c" not cleans  everything ?

           * when changing source layout, older derived files in the scons-out tree get
             left behind ... handle this by occasional rm -rf scons-out for a deep clean    

   == source ==
    
     http://prdownloads.sourceforge.net/scons/scons-1.3.0.tar.gz


   == realworld scons usage examples ==

     http://www.opensource.apple.com/source/JavaScriptCore/JavaScriptCore-525/JavaScriptCore.scons

EOU
}

scons-dir-pth(){
  local msg="=== $FUNCNAME : "
  local tmp=/tmp/$USER/env/$FUNCNAME/scons.pth && mkdir -p $(dirname $tmp)
  python-
  echo $(scons-dir) > $tmp
  echo $msg prepare $tmp to put scons-dir on sys path 
  cat $tmp
  local cmd="sudo cp $tmp $(python-site)/$(basename $tmp)"
  echo $msg $cmd
  eval $cmd 
}

scons-version(){ python -c "import SCons as _ ; print _.__version__ " ; } 

scons-dir-export(){ export SCONS_DIR=$(scons-dir) ;  }
scons-dir(){ 
   pkgr- 
   case $(pkgr-cmd) in 
       #port) echo /opt/local/lib/scons-1.2.0 ;; 
       port) echo /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/scons-2.0.1 ;; 
        yum) echo /usr/lib/scons       ;;
       ipkg) echo /opt/lib/scons-1.2.0       ;;
          *) echo /tmp ;;
   esac
}
scons-cd(){  cd $(scons-dir); }
scons-mate(){ mate $(scons-dir) ; }
scons-get(){
   #local dir=$(dirname $(scons-dir)) &&  mkdir -p $dir && cd $dir
   echo use your pkg manager to get scons ... see scons-usage
}





