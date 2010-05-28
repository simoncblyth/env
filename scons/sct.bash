# === func-gen- : scons/sct fgp scons/sct.bash fgn sct fgh scons
sct-src(){      echo scons/sct.bash ; }
sct-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sct-src)} ; }
sct-vi(){       vi $(sct-source) ; }
sct-env(){      
   elocal-  
   export SCT_DIR=$(sct-dir) 
}
sct-usage(){
  cat << EOU
     sct-src : $(sct-src)
     sct-dir : $(sct-dir)

 == Software Construction Toolkit ==

    Open Sourced Googles Extensions to SCons
        http://code.google.com/p/sct/

    sct 
         alias for sct-hammer 

    sct-hammer

 == Useful Options ==

     sct --help
         NB this is just the options provided by SCT, many more scons options
         can also be given to sct, see scons -H

     sct -c   :  cleans 
     sct -Q   :  omit the waffle

     sct --verbose
          see the commands issued

     sct --tree=derived
     sct --tree=all
          dependency handling descends into externals




 == Investigating Underpinnings == 

    sct-pth
       puts sct-site on sys path 

       BUT ... this fails to eliminate
       the need for hammer/wrapper invokation 
       add more hookup is done...

    sct-wrapper-spoof-
        commands for spoofing an SCT run from ipython commandline
        as an alternative to runninf the wrapper itself :

    In [1]: run /usr/local/env/scons/sct/wrapper.py --site-dir=/usr/local/env/scons/sct/site_scons
    scons: Reading SConscript files ...




EOU
}

sct-url(){ echo http://sct.googlecode.com/svn/trunk/ ; }
sct-dir(){ echo $(local-base)/env/scons/sct ; }
sct-cd(){  cd $(sct-dir); }
sct-mate(){ mate $(sct-dir) ; }
sct-get(){
   local dir=$(dirname $(sct-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d sct ] && svn co $(sct-url) sct
}

sct-pth(){
  local msg="=== $FUNCNAME : "
  local tmp=/tmp/env/$FUNCNAME/sct.pth && mkdir -p $(dirname $tmp)
  echo $(sct-dir)/site_scons > $tmp
  echo $msg prepare $tmp to put site_scons on sys path 
  cat $tmp
  local cmd="sudo cp $tmp $(python-site)/$(basename $tmp)"
  echo $msg $cmd
  eval $cmd
}

sct-hammer(){ $(sct-dir)/hammer.sh $* ; }
sct-wrapper-spoof-(){ cat << EOW
import sys
sys.argv.extend(['--site-dir=$(sct-dir)/site_scons','--file=main.scons'])
import SCons.Script
SCons.Script.main()
EOW
}

alias sct="sct-hammer"
