# === func-gen- : doc/sphinxcontrib fgp doc/sphinxcontrib.bash fgn sphinxcontrib fgh doc
sphinxcontrib-src(){      echo doc/sphinxcontrib.bash ; }
sphinxcontrib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sphinxcontrib-src)} ; }
sphinxcontrib-vi(){       vi $(sphinxcontrib-source) ; }
sphinxcontrib-env(){      elocal- ; }
sphinxcontrib-usage(){ cat << EOU

sphinx contrib
===============

feed
------

Features
~~~~~~~~

Building docs updates *~/w/_build/html/rss.xml* thus accessible at toplevel rss.xml


Installation
~~~~~~~~~~~~~

::

	simon:feed blyth$ sudo port select python python25
	simon:feed blyth$ sudo python setup.py install
	simon:feed blyth$ pbpaste > install.log
	simon:feed blyth$ pwd
	/usr/local/env/doc/sphinx-contrib/feed


503 Service Temporarily Unavailable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From *apache-logs* see that modjk is trying to pass along the request for rss.xml 
to jetty1, which is not running.

Have to comment in ``apache-edit``::

	# this catches rss.xml as used for sphinxcontrib.feed
	Include /data/heprez/install/apache/conf/heprez.conf

Note that:

#. URLs in the feed are incorrect for dirhtml style
#. nothing appears in latest


Config in w/conf.py
~~~~~~~~~~~~~~~~~~~~

::

    319 feed_base_url = "http://localhost/wdocs"
    320 feed_description = "The outpourings of a Sphinx developer"
    321 feed_filename = 'rss.xml'
    322 feed_title = 'Sphinx Syndicate Test Title'


EOU
}
sphinxcontrib-dir(){ echo $(local-base)/env/doc/sphinx-contrib ; }
sphinxcontrib-cd(){  cd $(sphinxcontrib-dir); }
sphinxcontrib-mate(){ mate $(sphinxcontrib-dir) ; }
sphinxcontrib-get(){
   local dir=$(dirname $(sphinxcontrib-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://bitbucket.org/birkenfeld/sphinx-contrib

}

sphinxcontrib-update(){

  sphinxcontrib-cd
  hg pull
  hg update
}
