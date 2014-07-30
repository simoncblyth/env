# === func-gen- : svn/pysvn fgp svn/pysvn.bash fgn pysvn fgh svn
pysvn-src(){      echo svn/pysvn.bash ; }
pysvn-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pysvn-src)} ; }
pysvn-vi(){       vi $(pysvn-source) ; }
pysvn-env(){      elocal- ; }
pysvn-usage(){ cat << EOU
PYSVN
========

* http://pysvn.tigris.org

Installs
---------

#. D, macports install (July 30, 2014)



Macports
----------




::

    delta:~ blyth$ port info py27-pysvn
    py27-pysvn @1.7.6_3 (python, devel)
    Variants:             universal

    Description:          The pysvn module is a python interface to the Subversion
                          version control system. This API exposes client interfaces
                          for managing a working copy, querying a repository, and
                          synchronizing the two.
    Homepage:             http://pysvn.tigris.org/

    Library Dependencies: python27, subversion
    Platforms:            darwin
    License:              Apache-1.1
    Maintainers:          blair@macports.org, yunzheng.hu@gmail.com,
                          openmaintainer@macports.org


::

    delta:~ blyth$ port contents py27-pysvn
    Port py27-pysvn contains:
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pysvn/__init__.py
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pysvn/__init__.py.template
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pysvn/_pysvn_2_7.so
      /opt/local/share/doc/py27-pysvn/examples/Client/parse_datetime.py
      /opt/local/share/doc/py27-pysvn/examples/Client/pysvn_print_doc.py
      /opt/local/share/doc/py27-pysvn/examples/Client/svn_cmd.py
      /opt/local/share/doc/py27-pysvn/generate_cpp_docs_from_html_docs.py
      /opt/local/share/doc/py27-pysvn/pysvn.html
      /opt/local/share/doc/py27-pysvn/pysvn_prog_guide.html
      /opt/local/share/doc/py27-pysvn/pysvn_prog_ref.html
      /opt/local/share/doc/py27-pysvn/pysvn_prog_ref.js






EOU
}
pysvn-dir(){ echo $(local-base)/env/svn/svn-pysvn ; }
pysvn-cd(){  cd $(pysvn-dir); }
pysvn-mate(){ mate $(pysvn-dir) ; }
pysvn-get(){
   local dir=$(dirname $(pysvn-dir)) &&  mkdir -p $dir && cd $dir

}

pysvn-docs(){
   open file:///opt/local/share/doc/py27-pysvn/pysvn_prog_guide.html
}


