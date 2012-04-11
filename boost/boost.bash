# === func-gen- : boost/boost fgp boost/boost.bash fgn boost fgh boost
boost-src(){      echo boost/boost.bash ; }
boost-source(){   echo ${BASH_SOURCE:-$(env-home)/$(boost-src)} ; }
boost-vi(){       vi $(boost-source) ; }
boost-env(){      elocal- ; }
boost-usage(){ cat << EOU

installed versions
~~~~~~~~~~~~~~~~~~~~
   
G  1.49.0
C  1.32.0-7.rhel4 
N  1.33.1

version history
~~~~~~~~~~~~~~~~

http://www.boost.org/users/history/



notable libs
~~~~~~~~~~~~~

file:///opt/local/share/doc/boost/doc/html/accumulators/user_s_guide.html
   probably a good start for re-implementing jima:avg in C++



installed documentation
~~~~~~~~~~~~~~~~~~~~~~~

  open file:///opt/local/share/doc/boost/doc/html/index.html

documentation system
~~~~~~~~~~~~~~~~~~~~~

https://svn.boost.org/trac/boost/wiki/BoostDocs/GettingStarted
https://svn.boost.org/trac/boost/wiki/DocsOrganization

sudo port install docbook-xml-4.2 docbook-xsl libxslt doxygen



EOU
}
boost-dir(){ echo $(local-base)/env/boost/boost-boost ; }
boost-cd(){  cd $(boost-dir); }
boost-mate(){ mate $(boost-dir) ; }
boost-get(){
   local dir=$(dirname $(boost-dir)) &&  mkdir -p $dir && cd $dir

}
