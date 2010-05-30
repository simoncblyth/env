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

 == Questions ==

   * how to publish a header ?
       * implemented INCLUDE_ROOT at top level and created a global 
         function to copy there using SCons Replicate 
             eg  EIncludes( env , ['path/to/header.h'] )

   * how to prevent tests from failing due to the needed libs not being present ?
       * some libs end up in the TEST_DIR but not all ... what is dictating this ?

   * in a SConscript or build.scons is the Imported env a clone
     or a reference to the "calling" env ? 
     OR should I be cloning inside the build.scons ? 





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
          note that dependency handling descends into externals

     sct --retest run_all_tests
     sct --retest run_large_tests       ## tests can be grouped 
     sct --retest run_test_cjsn

          the list of available tests to build/run is given by "sct --help"
          the "--retest" is needed as will usually say that test has run once already
          based on the existance of test output 


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
