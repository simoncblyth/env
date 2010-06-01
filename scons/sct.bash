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
       http://code.google.com/p/swtoolkit/
       http://code.google.com/p/swtoolkit/wiki/Introduction
       http://code.google.com/p/swtoolkit/wiki/Examples
       http://code.google.com/p/swtoolkit/wiki/Glossary
    Forum
       http://groups.google.com/group/swtoolkit

    Extends SCons via a site_scons dir
       http://www.scons.org/doc/HTML/scons-man.html


 == Installation / Running  ==

    sct-
    sct-get : SVN checkout the SCT sources

  Thats it : now find a "main.scons" to hammer ... eg 
     cd ~/env
     sct 

 == Functions == 

    sct        :  alias for sct-hammer 
    sct-hammer :  invoke the SCT hammer.sh script 

 == Installation Issues ==

  1) SCT unable to find SCons ...
{{{
    [blyth@cms01 e]$ sct
    Traceback (most recent call last):
    File "/data/env/local/env/scons/sct/wrapper.py", line 44, in <module>
       import SCons.Script
    ImportError: No module named SCons.Script
}}}
    Two ways to fix this :
       * define envvar SCONS_DIR using   : scons-dir-export
       * more robustly plant a .pth with : scons-dir-pth
         ... this only needs to be once to your python 


   2) Missing hammer.sh :
{{{
[blyth@cms02 e]$ sct
-bash: /data/env/local/env/scons/sct/hammer.sh: No such file or directory
}}}
      * You need to {{{sct- ; sct-get}}} first 



 == SCT/SCons Usage Questions ==

   * how to publish a header ?
       * implemented INCLUDE_ROOT at top level and created a global 
         function to copy there using SCons Replicate 
             eg  EIncludes( env , ['path/to/header.h'] )

       * possibly a better way, using ComponentPackage and Publish a header resource type 
           http://groups.google.com/group/swtoolkit/browse_thread/thread/0be8b7baf81d3527#


   * how to prevent tests from failing due to the needed libs not being present ?
       * some libs end up in the TEST_DIR but not all ... what is dictating this ?

       * http://code.google.com/p/swtoolkit/wiki/Glossary#ComponentTestProgram
          * says that libs get copies ... how are they determined
       * possibly a better way than setting library path ...
            http://groups.google.com/group/swtoolkit/browse_thread/thread/54f0fc28393cbabb#

   * how to set arguments for a test run ?
      * http://code.google.com/p/swtoolkit/wiki/Glossary#SetTargetProperty  may be the way  

   * in a SConscript or build.scons is the Imported env a clone
     or a reference to the "calling" env ? 
     OR should I be cloning inside the build.scons ? 



   *  why does "sct -c" not clean everything ?

      * when changing source layout, older derived files in the scons-out tree get
        left behind ... handle this by occasional rm -rf scons-out for a deep clean    

      * other derived files also not cleaned :

simon:e blyth$ find scons-out -type f
scons-out/.sconsign_darwin.dblite
scons-out/dbg/obj/aberdeen/AbtViz/AbtVizDict_Dict.h
scons-out/dbg/obj/aberdeen/DataModel/AbtDataModelDict_Dict.h
scons-out/dbg/obj/rootmq/rootmq_Dict.h

simon:e blyth$ find scons-out -type l
scons-out/dbg/lib/libAbtDataModel.so
scons-out/dbg/lib/libAbtViz.so
scons-out/dbg/lib/librootmq.so



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

sct-url(){ echo http://swtoolkit.googlecode.com/svn/trunk/ ; }
sct-dir(){ echo $(local-base)/env/scons/sct ; }
sct-cd(){  cd $(sct-dir); }
sct-mate(){ mate $(sct-dir) ; }
sct-get(){
   local dir=$(dirname $(sct-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d sct ] && svn co $(sct-url) sct
}
sct-hammer(){ $(sct-dir)/hammer.sh $* ; }
alias sct="sct-hammer"

sct-info(){ svn info $(sct-dir) ; }



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

sct-wrapper-spoof-(){ cat << EOW
import sys
sys.argv.extend(['--site-dir=$(sct-dir)/site_scons','--file=main.scons'])
import SCons.Script
SCons.Script.main()
EOW
}

