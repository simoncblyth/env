# === func-gen- : tools/fabric fgp tools/fabric.bash fgn fabric fgh tools
fabric-src(){      echo tools/fabric.bash ; }
fabric-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fabric-src)} ; }
fabric-vi(){       vi $(fabric-source) ; }
fabric-env(){      elocal- ; }
fabric-usage(){ cat << EOU

FABRIC
======


references
------------

http://docs.fabfile.org/en/1.4.2/index.html

http://awaseroot.wordpress.com/2012/04/25/fabric-tutorial-2-file-transfer-error-handling/

http://www.saltycrane.com/blog/2010/11/fabric-post-run-processing-python-decorator/

:e:`_docs/tools/fabric`


fabric observations
---------------------

bizarre non default
~~~~~~~~~~~~~~~~~~~~~

::

      env.use_ssh_config = True

http://stackoverflow.com/questions/3077281/pythons-fabric-connect-to-a-host-listed-ssh-config


non standard ports 
~~~~~~~~~~~~~~~~~~~~~

Seems the port must be in the host string it is not good enough to get the port 
via the SSH config.  Thus must do::

    fab -H Z9:229 hostname

Rather than::

    fab -H Z9 hostname

Related to 
* https://github.com/fabric/fabric/issues/138

flexible host setting
~~~~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/2326797/how-to-set-target-hosts-in-fabric-file 


alternatives
-------------

http://codespeak.net/execnet/index.html

http://tav.espians.com/fabric-python-with-cleaner-api-and-parallel-deployment-support.html


usage ideas 
-------------

network graph checking
~~~~~~~~~~~~~~~~~~~~~~~~

fabfile that invokes remote fabfiles that invokes further to 
test all ssh connections in network of nodes, pulling the results 
into rst format graphiz node diagram constructed in part on each node
and collected back to the invoking node

to stop the recursion getting out of control, need to handle 
the originating node differently OR use a dated output file that 
is checked for before proceeding to call buddies


installs
---------

C2
~~~

::

        [blyth@cms02 scm]$ which python
	/data/env/system/python/Python-2.5.6/bin/python
	[blyth@cms02 scm]$ which pip
	/data/env/system/python/Python-2.5.6/bin/pip
	[blyth@cms02 scm]$ pip install fabric

G
~~~

macports fabric is old  py25-fabric @0.1.1 so get 1.4.2 via  pip-2.5, 

:: 

       simon:~ blyth$ sudo pip-2.5 install Fabric


surprised where the fab went ``/opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin/fab``


EOU
}
fabric-dir(){ echo $(local-base)/env/tools/tools-fabric ; }
fabric-cd(){  cd $(fabric-dir) ;  }
fabric-scd(){  python- ; cd $(python-site)/fabric ;  }
fabric-mate(){ mate $(fabric-dir) ; }
fabric-get(){
   local dir=$(dirname $(fabric-dir)) &&  mkdir -p $dir && cd $dir

}

fabric-osx-ln(){
   cd /opt/local/bin
   sudo ln -s  /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin/fab fab
}

