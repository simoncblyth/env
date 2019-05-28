ipython-src(){ echo python/ipython.bash ; }
ipython-source(){  echo $(env-home)/$(ipython-src) ; }
ipython-vi(){      vi $(ipython-source) ; }


ipython-profile-path(){ echo ~/.ipython/profile_$1/ipython_config.py ; }
ipython-edit(){ vi $(ipython-profile-path $(ipython-profile)) ;} 
#ipython-profile(){ echo g4dae ; }
#ipython-profile(){ echo g4opticks ; }
ipython-profile(){ echo default ; }

ipython-nb(){
    chroma-
    ipython notebook --profile $(ipython-profile)
}


ipython-usage(){ cat << EOU


IPYTHON
========


jupyter
--------

* http://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/
* http://jakevdp.github.io/blog/2017/03/03/reproducible-data-analysis-in-jupyter/

ipython from conda
---------------------

Default profile dont work with python3, so make new one::

    epsilon:likelihood blyth$ ipython profile create foo
    epsilon:likelihood blyth$ ipython --profile=foo
    Python 3.7.0 (default, Jun 28 2018, 07:39:16) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

    IPython profile: foo

    In [1]: 


ipython HEP
-------------

* http://hep-ipython-tools.github.io

  calculation object to run jobs ?
  


debug ipython environment
---------------------------

::

    epsilon:~ blyth$ ipython --log-level=DEBUG 
    [TerminalIPythonApp] IPYTHONDIR set to: /Users/blyth/.ipython
    [TerminalIPythonApp] Using existing profile dir: u'/Users/blyth/.ipython/profile_default'
    [TerminalIPythonApp] Searching path [u'/Users/blyth', u'/Users/blyth/.ipython/profile_default', '/usr/local/etc/ipython', '/etc/ipython'] for config files
    [TerminalIPythonApp] Attempting to load config file: ipython_config.py
    [TerminalIPythonApp] Looking for ipython_config in /etc/ipython
    [TerminalIPythonApp] Looking for ipython_config in /usr/local/etc/ipython
    [TerminalIPythonApp] Looking for ipython_config in /Users/blyth/.ipython/profile_default
    [TerminalIPythonApp] Loaded config file: /Users/blyth/.ipython/profile_default/ipython_config.py
    [TerminalIPythonApp] Looking for ipython_config in /Users/blyth

    [TerminalIPythonApp] Loading IPython extensions...
    [TerminalIPythonApp] Loading IPython extension: storemagic
    [TerminalIPythonApp] Running code from IPythonApp.exec_lines...
    [TerminalIPythonApp] Running code in user namespace: 
    [TerminalIPythonApp] Running code in user namespace: import os, sys, logging
    [TerminalIPythonApp] Running code in user namespace: log = logging.getLogger(__name__)
    [TerminalIPythonApp] Running code in user namespace: import numpy as np
    [TerminalIPythonApp] Running code in user namespace: import matplotlib.pyplot as plt
    [TerminalIPythonApp] Running code in user namespace: from mpl_toolkits.mplot3d import Axes3D
    [TerminalIPythonApp] Running code in user namespace: 
    [TerminalIPythonApp] Running code in user namespace: sys.path.append(os.path.expanduser("~"))
    [TerminalIPythonApp] Running code in user namespace: 
    [TerminalIPythonApp] Running code in user namespace: from opticks.ana.base import opticks_main
    [TerminalIPythonApp] Running code in user namespace: ok = opticks_main()
    args: /opt/local/bin/ipython --log-level=DEBUG
    [TerminalIPythonApp] Running code in user namespace: 
    [TerminalIPythonApp] Running code in user namespace: from opticks.ana.histype import HisType
    [TerminalIPythonApp] Running code in user namespace: 
    [TerminalIPythonApp] Running code in user namespace: #np.set_printoptions(suppress=True, precision=3)
    [TerminalIPythonApp] Running code in user namespace: # sqlite3 database querying into ndarray
    [TerminalIPythonApp] Running code in user namespace: #from  _npar import npar as q 
    [TerminalIPythonApp] Running code in user namespace: #logging.basicConfig(level=logging.INFO)
    [TerminalIPythonApp] Running code in user namespace: 
    [TerminalIPythonApp] Starting IPython's mainloop...

    In [1]:    



profiles
---------

::

    (chroma_env)delta:~ blyth$ ipython profile create g4dae
    [ProfileCreate] Generating default config file: u'/Users/blyth/.ipython/profile_g4dae/ipython_config.py'
    [ProfileCreate] Generating default config file: u'/Users/blyth/.ipython/profile_g4dae/ipython_notebook_config.py'
    [ProfileCreate] Generating default config file: u'/Users/blyth/.ipython/profile_g4dae/ipython_nbconvert_config.py'
    (chroma_env)delta:~ blyth$ 



Careful where to put the -- 
------------------------------

When out front get error::

    delta:wtracdb blyth$ wtracdb-i
    INFO:env.sqlite.db:opening /usr/local/workflow/sysadmin/wtracdb/workflow/db/trac.db 
    args: /opt/local/bin/ipython -i -- [u'system', u'permission', u'auth_cookie', u'session', u'session_attribute', u'attachment', u'wiki', u'revision', u'node_change', u'ticket', u'ticket_change', u'ticket_custom', u'enum', u'component', u'milestone', u'version', u'report', u'tags', u'bitten_config', u'bitten_platform', u'bitten_rule', u'bitten_build', u'bitten_slave', u'bitten_step', u'bitten_error', u'bitten_log', u'bitten_log_message', u'bitten_report', u'bitten_report_item', u'fullblog_posts', u'fullblog_comments']
    [TerminalIPythonApp] WARNING | File not found: u"[u'system',"

Different invokation for interactive python and ipython::

     88 wtracdb-s(){ sqlite3 $(wtracdb-path) ; }
     89 wtracdb-p(){ python -i $(workflow-home)/sysadmin/wtracdb/wtracdb.py $(wtracdb-path) ; }
     90 wtracdb-i(){ ipython -i $(workflow-home)/sysadmin/wtracdb/wtracdb.py -- $(wtracdb-path) ; }



to avoid having to specifiy profiles
-------------------------------------

::

    simon:.ipython blyth$ mv profile_default profile_default_orig
    simon:.ipython blyth$ mv profile_g4opticks profile_default
    simon:.ipython blyth$ 

    simon:cfg4 blyth$ opticks-find profile=g4opticks
    ./ana/ana.bash:    args: /opt/local/bin/ipython --profile=g4opticks
    ./tests/tconcentric.bash:tconcentric-i(){     ipython --profile=g4opticks -i $(which tconcentric.py) --  $(tconcentric-args) $* ; } 
    ./tests/tconcentric.bash:tconcentric-d(){     ipython --profile=g4opticks -i $(which tconcentric_distrib.py) --  $(tconcentric-args) $* ; } 
    ./tests/tg4gun.bash:tg4gun-i(){     ipython --profile=g4opticks -i $(which g4gun.py) --  $(tg4gun-args) $* ; }
    ./tests/tgltf.bash:tgltf-rip(){ local fnpy=$1 ; local py=$TMP/$fnpy.py ; $fnpy > $py ;  ipython --profile=g4opticks -i $py ; }
    simon:opticks blyth$ 
    simon:opticks blyth$ 
    simon:opticks blyth$ vi ana/ana.bash
    simon:opticks blyth$ vi tests/tconcentric.bash
    simon:opticks blyth$ vi tests/tg4gun.bash
    simon:opticks blyth$ vi tests/tgltf.bash
    simon:opticks blyth$ 
    simon:opticks blyth$ 
    simon:opticks blyth$ opticks-find profile=g4opticks
    simon:opticks blyth$ 


debugging
-----------


1. Place an assert where you want a backtrace from.

2. Start ipthon with --pdb option

3. run ~/opticks/ana/ckm.py   ## doesnt get things from PATH 


Or more quickly::

   [blyth@localhost 1]$ ipython --pdb ~/opticks/ana/ckm.py
 
Or automate that a bit:: 

   ip(){ local py=$1 ; shift ; ipython --pdb $(which $py) $* ; }  ## in profile

ip ckm.py:: 

    [blyth@localhost 1]$ ip ckm.py
    args: /home/blyth/opticks/ana/ckm.py
    [2019-05-28 11:43:18,095] p454944 {/home/blyth/opticks/ana/base.py:332} INFO -  ( opticks_environment
    [2019-05-28 11:43:18,095] p454944 {/home/blyth/opticks/ana/base.py:337} INFO -  ) opticks_environment
    [2019-05-28 11:43:18,097] p454944 {/home/blyth/opticks/ana/base.py:631} WARNING - failed to load json from $OPTICKS_DATA_DIR/resource/GFlags/abbrev.json
    ---------------------------------------------------------------------------
    AssertionError                            Traceback (most recent call last)
    /home/blyth/opticks/ana/ckm.py in <module>()
         18     np.set_printoptions(suppress=True, precision=3)
         19 
    ---> 20     evt = Evt(tag=args.tag, src=args.src, det=args.det, seqs=[], args=args)
         21 
         22     log.debug("evt")

    /home/blyth/opticks/ana/evt.pyc in __init__(self, tag, src, det, args, maxrec, rec, dbg, label, seqs, not_, nom, smry)
        205            return
        206 
    --> 207         self.init_types()
        208         self.init_gensteps(tag, src, det, dbg)
        209         self.init_photons()

    /home/blyth/opticks/ana/evt.pyc in init_types(self)
        234         """
        235         log.debug("init_types")
    --> 236         self.hismask = HisMask()
        237         self.histype = HisType()
        238 

    /home/blyth/opticks/ana/hismask.pyc in __init__(self)
         21         log.debug("HisMask.__init__")
         22         flags = EnumFlags()
    ---> 23         abbrev = Abbrev("$OPTICKS_DATA_DIR/resource/GFlags/abbrev.json")
         24         MaskType.__init__(self, flags, abbrev)
         25         log.debug("HisMask.__init__ DONE")

    /home/blyth/opticks/ana/base.pyc in __init__(self, path)
        663     """
        664     def __init__(self, path):
    --> 665         js = json_(path)
        666 
        667         names = map(str,js.keys())

    /home/blyth/opticks/ana/base.pyc in json_(path)
        630     except IOError:
        631         log.warning("failed to load json from %s" % path)
    --> 632         assert 0
        633         _json[path] = {}
        634     pass

    AssertionError: 
    > /home/blyth/opticks/ana/base.py(632)json_()
        630     except IOError:
        631         log.warning("failed to load json from %s" % path)
    --> 632         assert 0
        633         _json[path] = {}
        634     pass

    ipdb> 


Commands::

    ipdb> help a
    a(rgs)
    Print the arguments of the current function.
    ipdb> a
    self = CF(1,torch,concentric,['TO BT BT BT BT SA']) 
    qwn = X
    irec = 1
    ipdb> up
    > /Users/blyth/opticks/ana/cfplot.py(126)qwns_plot()
        125 
    --> 126         rqwn_bins, aval, bval, labels = scf.rqwn(qwn, irec)
        127 



ipython bash
---------------

* https://stackoverflow.com/questions/15927142/execute-bash-command-from-ipython

* https://ipython.org/ipython-doc/3/interactive/magics.html

via file is easier
~~~~~~~~~~~~~~~~~~~~

::

    rip(){ local fnpy=$1 ; local py=$TMP/$fnpy.py ; $fnpy > $py ;  ipython --profile=g4opticks -i $py ; }

    simon:opticks blyth$ rip tgltf-gdml--     # start ipython with the python script produced by the argument bash function




Cell magics
~~~~~~~~~~~~~

::

    %%bash
    %%bash script magic

    Run cells with bash in a subprocess.

    This is a shortcut for %%script bash


::

    In [6]: %%bash
       ...: source ~/.bash_profile
       ...: tgltf-
       ...: tgltf-gdml--     ## bash function that pipes some python
       ...: 

    import os, logging, sys, numpy as np
    log = logging.getLogger(__name__)
    ...


    In [9]: %%bash --out b
    source ~/.bash_profile ; tgltf- ; tgltf-gdml--
       ...: 

    In [10]: print b

    import os, logging, sys, numpy as np

    log = logging.getLogger(__name__)





plotly
-------

* https://plot.ly/python/3d-plots-tutorial/

notebooks web interface
-------------------------

* http://ipython.org/ipython-doc/2/notebook/index.html

* http://nbviewer.ipython.org
 
  * many examples of note book usage 

* http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb

  * matplotlib inline 

* http://nbviewer.ipython.org/github/ipython/ipython/blob/2.x/examples/Notebook/User%20Interface.ipynb

  * shortcut keys 

* http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut5.html




nbviewer
---------

* http://nbviewer.ipython.org/faq
* https://github.com/ipython/nbviewer


customize profile
------------------

::

    ipython-edit

::

    exec_lines = r"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    ph = lambda _:np.load(os.environ['DAE_PATH_TEMPLATE'] % _)
    np.set_printoptions(suppress=True, precision=3)
    """
    c.InteractiveShellApp.exec_lines = exec_lines.split("\n")





installs
----------

D
~~

Uninstall macports ipython and pip install inside chroma virtualenv, as misbehaves there otherwise::

    delta:~ blyth$ sudo port uninstall py27-ipython
    Password:
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Deactivating py27-ipython @1.1.0_1+scientific
    --->  Cleaning py27-ipython
    --->  Uninstalling py27-ipython @1.1.0_1+scientific
    --->  Cleaning py27-ipython
    delta:~ blyth$ 
    delta:~ blyth$ which ipython
    delta:~ blyth$ chroma-
    (chroma_env)delta:~ blyth$ which pip
    /usr/local/env/chroma_env/bin/pip
    (chroma_env)delta:~ blyth$ pip install ipython
    Downloading/unpacking ipython
      Downloading ipython-1.2.1.tar.gz (8.7MB): 8.7MB downloaded
      Running setup.py egg_info for package ipython
        
    Installing collected packages: ipython
      Running setup.py install for ipython
        
        Installing ipcontroller script to /usr/local/env/chroma_env/bin
        Installing iptest script to /usr/local/env/chroma_env/bin
        Installing ipcluster script to /usr/local/env/chroma_env/bin
        Installing ipython script to /usr/local/env/chroma_env/bin
        Installing pycolor script to /usr/local/env/chroma_env/bin
        Installing iplogger script to /usr/local/env/chroma_env/bin
        Installing irunner script to /usr/local/env/chroma_env/bin
        Installing ipengine script to /usr/local/env/chroma_env/bin
    Successfully installed ipython
    Cleaning up...
    (chroma_env)delta:~ blyth$ 


ipython libedit issue, at ipython startup message::

    It is highly recommended that you install readline, which is easy_installable:
         easy_install readline
    Note that `pip install readline` generally DOES NOT WORK, because
    it installs to site-packages, which come *after* lib-dynload in sys.path,
    where readline is located.  It must be `easy_install readline`, or to a custom
    location on your PYTHONPATH (even --user comes after lib-dyload).

do so::

    (chroma_env)delta:~ blyth$ easy_install readline
    ...
    Running readline-6.2.4.1/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-t49Iq7/readline-6.2.4.1/egg-dist-tmp-DcEEyu
    ld: warning: ignoring file /opt/local/lib/libncurses.dylib, file was built for x86_64 which is not the architecture being linked (i386): /opt/local/lib/libncurses.dylib
    Adding readline 6.2.4.1 to easy-install.pth file

    Installed /usr/local/env/chroma_env/lib/python2.7/site-packages/readline-6.2.4.1-py2.7-macosx-10.9-x86_64.egg


refs
-------


  Good intro to pylab/numpy/ipython etc...
     http://conference.scipy.org/scipy2010/tutorials.html



   Issue with macports ipython 2.5 and readline, getting gibberish prompt
   1st try uninstall and install 

simon:qxml blyth$ sudo port uninstall py25-ipython
--->  The following versions of py25-ipython are currently installed:
--->      py25-ipython @0.9.1_0
--->      py25-ipython @0.10.2_1 (active)


    sudo port clean py25-ipython   
    sudo port install py25-ipython   -scientific

To make the Python 2.5 version of IPython the one that is run
    when you execute the commands without a version suffix, e.g. 'ipython',
        run:
	        port select --set ipython ipython25

		--->  Cleaning py25-ipython


uninstallation/installation of py25-readline + py25-ipython 
still gives a gibberized prompt...

make it less objectionable by changing config to use "colors NoColor"

   vi ~/.ipython/ipythonrc




IHEP with local python 2.5.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Later versions of ipython require py2.6::

	[dayabay] /home/blyth > rm -rf build
	[dayabay] /home/blyth > pip install ipython==0.10
	Downloading/unpacking ipython==0.10
	  Downloading ipython-0.10.tar.gz (5.8Mb): 5.8Mb downloaded
	  Running setup.py egg_info for package ipython
	Installing collected packages: ipython
	  Running setup.py install for ipython
	    Installing iptest script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipythonx script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipcluster script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipython script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing pycolor script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipcontroller script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipengine script to /home/blyth/local/python/Python-2.5.6/bin
	Successfully installed ipython
	Cleaning up...
	[dayabay] /home/blyth > 






EOU
}


ipython-env(){
  elocal-
}


ipython-version(){ ipython -V ; }


ipython-get(){


  local msg="=== $FUNCNAME "

  local nik=ipython
  #local nam=$nik-0.8.1
  local nam=$nik-0.8.2
  local tgz=$nam.tar.gz
  local url=http://ipython.scipy.org/dist/$tgz

  local dir=$LOCAL_BASE/python/$nik   
   
   echo $msg 
   
  mkdir -p $dir || return 1
  cd $dir
  
  test -f $tgz || curl -o $tgz $url
  test -d $nam || tar zxvf $tgz
 
  cd $nam
  
 # 
 # unix python OR MacPython 
 #
 # local py=python
 #  local py="sudo /usr/local/bin/python"
 
  echo $msg installing into the python in your path $(which python) ===
  which python 
 
   python -c "import sys;print sys.prefix"
   $SUDO python setup.py install
 
 #
 #  this simple python switch , isnt working tis sensitive to the environment ... so 
 #
 
 
  
 
 
}

ipython-rm(){

  local dir=$(dirname $(which ipython))
  

}



ipython-check(){

    local p=$(which python)
    local i=$(which ipython)
    
    [ "$(dirname $p)" == "$(dirname $i)" ] && echo "1" || echo "0" 
}

ipython-fix(){

   local chk=$(ipython-check)
   if [ $chk == 1 ]; then
       local i=$(which ipython)
       echo === ipython-fix editing the ipython script $i 
       perl -i.orig -pe '$. == 1 && s/#!.*/#!\/usr\/bin\/env python/; '  $i
       diff $i.orig $i
    else
       echo === ipython-fix paths to python and ipython must be from the same folder 
    fi



}



ipython-readline(){

   # recipe from http://ipython.scipy.org/moin/InstallationOSXLeopard 
   # BUT turns out that the egg is in pypi already so can use the ez solution
   #     http://pypi.python.org/pypi/readline/2.5.1
   #

   local iwd=$PWD
   local dir=$LOCAL_BASE/env/ipython && mkdir -p $dir
    cd $dir
   
   local nam=python-readline-leopard
   local tgz=$nam-011808.tar.gz
   local url=http://ipython.scipy.org/moin/InstallationOSXLeopard?action=AttachFile\&do=get\&target=$tgz
  
   [ ! -f $tgz ] && curl -o $tgz $url
   [ ! -d $nam ] && tar zxvf $tgz
   
   cd $nam
   tgz=readline-5.2.tar.gz
   url=http://ftp.gnu.org/gnu/readline/$tgz
   
   egg=readline-2.5.1-py2.5-macosx-10.5-fat.egg
   
   [ ! -f $tgz ] && curl -o $tgz $url
   [ ! -f $egg ] && ./build.sh

   #cd $iwd
}

ipython-readline-ez(){
   easy_install readline==2.5.1
}


ipython-easyinstall-log(){

   # hmm seem to have double egged the pudding ?

cat << EOL

 easy_install readline-2.5.1-py2.5-macosx-10.5-fat.egg 
Processing readline-2.5.1-py2.5-macosx-10.5-fat.egg
creating /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-fat.egg
Extracting readline-2.5.1-py2.5-macosx-10.5-fat.egg to /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages
Adding readline 2.5.1 to easy-install.pth file

Installed /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-fat.egg
Processing dependencies for readline==2.5.1
Searching for readline==2.5.1
Reading http://pypi.python.org/simple/readline/
Reading http://www.python.org/
Best match: readline 2.5.1
Downloading http://pypi.python.org/packages/2.5/r/readline/readline-2.5.1-py2.5-macosx-10.5-ppc.egg#md5=25ebe33023a003c8bb8ba7507944f29c
Processing readline-2.5.1-py2.5-macosx-10.5-ppc.egg
creating /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-ppc.egg
Extracting readline-2.5.1-py2.5-macosx-10.5-ppc.egg to /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages
Removing readline 2.5.1 from easy-install.pth file
Adding readline 2.5.1 to easy-install.pth file

Installed /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-ppc.egg
Finished processing dependencies for readline==2.5.1

EOL

}





# WARNING: Readline services not available on this platform.
# WARNING: The auto-indent feature requires the readline library
# WARNING: Proper color support under MS Windows requires the pyreadline library.
# You can find it at:
# http://ipython.scipy.org/moin/PyReadline/Intro
# Gary's readline needs the ctypes module, from:
# http://starship.python.net/crew/theller/ctypes
# (Note that ctypes is already part of Python versions 2.5 and newer).
#
# Defaulting color scheme to 'NoColor'
# Python 2.5.1 (r251:54863, May  8 2007, 22:27:26) 
# Type "copyright", "credits" or "license" for more information.
# 
# IPython 0.8.1 -- An enhanced Interactive Python.
# ?       -> Introduction to IPython's features.
# %magic  -> Information about IPython's 'magic' % functions.
# help    -> Python's own help system.
# object? -> Details about 'object'. ?object also works, ?? prints more.
#
