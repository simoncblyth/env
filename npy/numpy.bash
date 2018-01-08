# === func-gen- : npy/numpy fgp npy/numpy.bash fgn numpy fgh npy
numpy-src(){      echo npy/numpy.bash ; }
numpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(numpy-src)} ; }
numpy-vi(){       vi $(numpy-source) ; }
numpy-env(){      elocal- ; }
numpy-usage(){
  cat << EOU





* https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to

New in 1.10.0

::

    In [43]: np.broadcast_to( a[0], (len(a), 6, 4))
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    <ipython-input-43-b60b43e7901f> in <module>()
    ----> 1 np.broadcast_to( a[0], (len(a), 6, 4))

    AttributeError: 'module' object has no attribute 'broadcast_to'

    In [44]: np.__version__
    Out[44]: '1.9.2'








#. numpy 1.8.0 requires py26+

::

    [blyth@belle7 pycollada]$ sudo bash -c "LD_LIBRARY_PATH=$LD_LIBRARY_PATH easy_install numpy==1.7.1" 
    Searching for numpy==1.7.1
    Reading http://pypi.python.org/simple/numpy/
    Best match: numpy 1.7.1
    Downloading https://pypi.python.org/packages/source/n/numpy/numpy-1.7.1.zip#md5=9a72db3cad7a6286c0d22ee43ad9bc6c
    Processing numpy-1.7.1.zip
    Running numpy-1.7.1/setup.py -q bdist_egg --dist-dir /tmp/easy_install-8lHRyq/numpy-1.7.1/egg-dist-tmp-cbtX4y
    Running from numpy source directory.
    error: SandboxViolation: open('/dev/null', 'w') {}

    The package setup script has attempted to modify files on your system
    that are not within the EasyInstall build area, and has been aborted.

    This package cannot be safely installed by EasyInstall, and may not
    support alternate installation locations even if you run its setup
    script by hand.  Please inform the package's author and the EasyInstall
    maintainers to find out if a fix or workaround is available.
    [blyth@belle7 pycollada]$ 





     numpy-src : $(numpy-src)
     numpy-dir : $(numpy-dir)
     
       http://github.com/numpy/numpy
       http://projects.scipy.org/numpy/report/6?asc=1&sort=modified&USER=anonymous

       http://www.scipy.org/
            svn co http://svn.scipy.org/svn/scipy/trunk scipy


   == references ==

       https://github.com/pv/advanced-numpy-tutorial
       http://www.scipy.org/Cookbook

       http://belle7.nuu.edu.tw/numpy/
       http://belle7.nuu.edu.tw/advanced-numpy-tutorial/

    == numpy documentation ==

       numpy-doc  
            build the sphinx based documentation
                (on N latex problem : preview.sty not found )        

       numpy-nginx
            plant a link from nginx-htdocs...  check results at 
                http://belle7.nuu.edu.tw/numpy/

    == debug build ==

       numpy-cd ; rm -rf build ; numpy-build --debug 
            http://projects.scipy.org/numpy/ticket/539

    == N : npy virtual py24 ==
 
         system python 2.4 has a yum installed numpy 1.1
         experiment with more recent numpy in virtual python : vip-npy 

    == C : in source py25  ==

       hostname   : cms01.phys.ntu.edu.tw
       version    : 2.0.0.dev-cfd4c05
       installdir : /data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy

      see vip- for installation via pip into virtual python env 

    == G : (macports) python_select python27 ==

      Numpy installs into :
        /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/

   == Fork numpy on github  ==

     Follow http://help.github.com/forking/
     clone the fork with 
   
         git clone git@github.com:scb-/numpy.git
         cd numpy
         git remote add upstream git://github.com/numpy/numpy.git
         git fetch upstream

    Thence can make changes and locally commut changes to my numpy fork 
    and push to (my) master on github with :

         git push origin master

    After which can issue   http://help.github.com/pull-requests/


   == rejig numpy fork to follow numpy dev workflow  ... as i forgot to make an initial branch ==

      http://cms01.phys.ntu.edu.tw/np/dev/gitwash/patching.html
          guidlines start by branching ... and do all changes one the branch 

        1) github web UI/Admin : delete scb-/numpy 
        2) github web UI numpy/numpy ... fork again (had to reload a few times to not see my datetime fix) 

        3) move the old repo aside, and clone the fresh one :

         [blyth@cms01 npy]$ mv numpy numpy_0
         [blyth@cms01 npy]$ git clone git@github.com:scb-/numpy.git
         Initialized empty Git repository in /data/env/local/env/npy/numpy/.git/
         Enter passphrase for key '/home/blyth/.ssh/id_rsa': 


         4) follow the github forking help http://help.github.com/forking/

           cd numpy
           git remote add upstream git://github.com/numpy/numpy.git
           [blyth@cms01 numpy]$ git remote
           origin
           upstream 

           [blyth@cms01 numpy]$ git fetch upstream
           ...
           * refs/remotes/upstream/maintenance/1.5.x: storing branch 'maintenance/1.5.x' of git://github.com/numpy/numpy
             commit: f3d4e73
           * refs/remotes/upstream/master: storing branch 'master' of git://github.com/numpy/numpy
             commit: 147f817


        5) this time immediately make a branch for each fix i have in mind  

             http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html#making-a-new-feature-branch

             git checkout master    ## move off the branch to allow to delete
             git branch -D datetime-s-hours-more-than-24-fix
  
             git branch datetime64s-hours-more-than-24
             git checkout datetime64s-hours-more-than-24

            Grab the mods from the repo are about to abandon and put into the named branch in new fork  ...

             cd ..

            [blyth@cms01 npy]$ diff --recursive --brief numpy_0 numpy | grep differ | grep -v .git 
            Files numpy_0/numpy/core/src/multiarray/datetime.c and numpy/numpy/core/src/multiarray/datetime.c differ
            Files numpy_0/numpy/core/tests/test_datetime.py and numpy/numpy/core/tests/test_datetime.py differ

            [blyth@cms01 numpy]$ git commit -a -m "fix for datetime64[s] hours exceeding 24 and test to demonstrate  "
            Created commit fd18c43: fix for datetime64[s] hours exceeding 24 and test to demonstrate
            2 files changed, 7 insertions(+), 1 deletions(-)

            [blyth@cms01 numpy]$ git --version
            git version 1.5.2.1
            simon:numpy blyth$ git --version
            git version 1.7.2.2
             
            cms01 git is too old for ...
                 git push --set-upstream origin datetime64s-hours-more-than-24
            so just do ..
                 git push origin datetime64s-hours-more-than-24

            this is visible on github when switch to the branch 
                 https://github.com/scb-/numpy/tree/datetime64s-hours-more-than-24


            make pull request from this branch
                 http://help.github.com/pull-requests/

            in shows up in the numpy pulls, not my ones
                 https://github.com/numpy/numpy/pulls

 

   == Unsure if still need for this change .. ==

   Was triggering segv in doing repr of arrays derived from buffers ...
   need to capture issue in test 

{{{
[blyth@cms01 tests]$ git diff
diff --git a/numpy/core/src/multiarray/scalarapi.c b/numpy/core/src/multiarray/scalarapi.c
index 87e140c..0f84d87 100644
--- a/numpy/core/src/multiarray/scalarapi.c
+++ b/numpy/core/src/multiarray/scalarapi.c
@@ -674,7 +674,7 @@ PyArray_Scalar(void *data, PyArray_Descr *descr, PyObject *base)
         memcpy(&(((PyDatetimeScalarObject *)obj)->obmeta), dt_data,
                sizeof(PyArray_DatetimeMetaData));
     }
-    if (PyTypeNum_ISFLEXIBLE(type_num)) {
+    if (PyTypeNum_ISEXTENDED(type_num)) {
         if (type_num == PyArray_STRING) {
             destptr = PyString_AS_STRING(obj);
             ((PyStringObject *)obj)->ob_shash = -1;

}}}



     == references ==

         http://efreedom.com/Question/1-877578/Fastest-Way-Convert-Numpy-Array-Sparse-Dictionary


     == pip source installation of branch ==

          pip install -e git+git@github.com:scb-/numpy.git@datetime64s-hours-more-than-24#egg=numpy 
               ## pushable clone requires passphrase for key 

          pip install -e git+git://github.com/scb-/numpy.git@datetime64s-hours-more-than-24#egg=numpy 
               ## un-pushable clone






EOU
}
numpy-dir(){ 
   ## virtualized takes precedent 
   [ -n "$VIRTUAL_ENV" ] && echo $VIRTUAL_ENV/src/numpy && return 
   echo $(local-base)/env/npy/$(numpy-name) ; 
}
numpy-cd(){  cd $(numpy-dir)/$1; }
numpy-scd(){  cd $(env-home)/npy/numpy/$1; }
numpy-mate(){ mate $(numpy-dir) ; }

#numpy-name(){ echo upstream ; }
numpy-name(){ 
    case $USER in
      blyth) echo scbfork ;;
      thho) echo scbfork_ro ;; 
    esac
}

numpy-url(){
   case $(numpy-name) in 
      upstream) echo git://github.com/numpy/numpy.git ;;
       scbfork) echo git@github.com:scb-/numpy.git    ;;
    scbfork_ro) echo git://github.com/scb-/numpy.git  ;;
   esac
}


numpy-tute(){
   cd $(local-base)/env/npy/ ; 

   local name=Euroscipy-intro-tutorials
   [ ! -d "$name" ] &&  git clone https://github.com/scipy-lectures/$name.git

   cd $name/lecture_notes
   make html

   ## hmmm needs : scipy.interpolate
   

# git clone https://github.com/numpy/numpy.git numpy

numpy-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir
   [ -d numpy ] && echo numpy exists already, remove before re-cloneing  && return 1 
   git clone $(numpy-url) $(numpy-name)
   numpy-cd
   git branch 
}
numpy-wipe(){
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir
   rm -rf numpy
}

numpy-version(){    local iwd=$PWD ; cd /tmp ; python -c "import numpy as np ; print np.__version__ " ;  cd $iwd ; }
numpy-installdir(){ python -c "import os, numpy as np ; print os.path.dirname(np.__file__) " ; }
numpy-include(){    python -c "import numpy as np ; print np.get_include() " ; }
numpy-info(){  cat << EOI  
    hostname   : $(hostname)
    version    : $(numpy-version)
    installdir : $(numpy-installdir)
    include    : $(numpy-include)

EOI
}

numpy-doc-preq(){
    pip install -U sphinx
}
numpy-doc(){
    numpy-cd doc
   [ "$(which sphinx-build)" == "" ] && echo $msg install sphinx first && return 
    make html
}
numpy-nginx(){
   nginx-
   local docd=$(numpy-dir)/doc/build/html/ 
   [ ! -d "$docd" ] && echo $msg ERROR no docs dir at  $docd && return
   nginx-ln $docd numpy
}



numpy-build(){
   numpy-cd
   which python
   python setup.py build $*
}

numpy-install(){
   numpy-cd
   which python
   ## using virtual python environments / or source python avoids sudo hassles  ...
   local cmd="python setup.py install"
   echo $msg $cmd
   eval $cmd
}

numpy-sinstall(){
   numpy-cd
   which python
   ## using virtual python environments / or source python avoids sudo hassles  ...
   local cmd="sudo python setup.py install"
   echo $msg $cmd
   eval $cmd
}


numpy-test(){
   local iwd=$PWD
   cd /tmp
   python -c 'import numpy; numpy.test()'
   cd $iwd 
}


