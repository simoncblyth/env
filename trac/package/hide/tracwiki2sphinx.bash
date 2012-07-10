tracwiki2sphinx-vi(){ vi $BASH_SOURCE ; }
tracwiki2sphinx-usage(){ cat << EOU

TracWiki2Sphinx
=================

Set up home for tracwiki2sphinx http://dayabay.phys.ntu.edu.tw/repos/tracdev/tracwiki2sphinx/trunk
starting with::

        svn co -N http://dayabay.phys.ntu.edu.tw/repos/tracdev/

Following egg install::

	g4pb-2:trunk blyth$ sudo /usr/bin/python setup.py install
	creating TracWiki2Sphinx.egg-info
	...
	Installed /Library/Python/2.5/site-packages/TracWiki2Sphinx-0.0.1-py2.5.egg
	Processing dependencies for TracWiki2Sphinx==0.0.1
	Finished processing dependencies for TracWiki2Sphinx==0.0.1

And enabling the component,  a "Sphinx RST" link appears at the page foot:

  * http://localhost/tracs/workflow/wiki/SynologyBonjour?format=spx

Reference
-----------

 * http://dayabay.phys.ntu.edu.tw/tracs/tracdev/timeline/
 * http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html
 * http://docutils.sourceforge.net/docs/ref/rst/roles.html
 * http://sphinx.pocoo.org/rest.html
 * http://sphinx.pocoo.org/markup/inline.html#inline-markup

 * http://trac-hacks.org/browser/downloadsplugin/0.11/tracdownloads/tags.py

     * for example of usage of TracTags by another plugin


 * /usr/local/env/trac/package/tractags/trunk/tractags





Testing
-------

  * http://localhost/tracs/workflow/wiki/TracWiki2Sphinx


More convenient development with egg link
------------------------------------------

Avoid repeated re-installation using egg link

::

	g4pb-2:trunk blyth$ sudo /usr/bin/python setup.py develop
	...
	Creating /Library/Python/2.5/site-packages/TracWiki2Sphinx.egg-link (link to .)
	Removing TracWiki2Sphinx 0.0.1 from easy-install.pth file
	Adding TracWiki2Sphinx 0.0.1 to easy-install.pth file

	Installed /Users/blyth/tracdev/tracwiki2sphinx/trunk
	Processing dependencies for TracWiki2Sphinx==0.0.1
	Finished processing dependencies for TracWiki2Sphinx==0.0.1


encoding issue for RST links from  some pages 
----------------------------------------------

::

   tracwiki2sphinx PARSER ERROR: 'charmap' codec can't encode character u'\xa6' in position 9154: character maps to




EOU

}

tracwiki2sphinx-env(){
  elocal-
  package-
  trac-
  export TRAC2SPHINX_BRANCH=trunk
}

tracwiki2sphinx-package(){ echo tracwiki2sphinx ; }
tracwiki2sphinx-revision(){ echo HEAD ;  }
tracwiki2sphinx-url(){     
   trac-
   echo $(trac-localserver)/repos/tracdev/tracwiki2sphinx/$(tracwiki2sphinx-branch) 
}

tracwiki2sphinx-reldir(){
   echo -n 
}
tracwiki2sphinx-fix(){
   echo -n
}
tracwiki2sphinx-prepare(){
   tracwiki2sphinx-enable $*
}

tracwiki2sphinx-dir(){ echo $HOME/tracdev/tracwiki2sphinx/trunk/ ; }
tracwiki2sphinx-cd(){ cd $(tracwiki2sphinx-dir) ; }
tracwiki2sphinx-install(){
   tracwiki2sphinx-cd
   #sudo /usr/bin/python setup.py install
   sudo /usr/bin/python setup.py develop
}


tracwiki2sphinx-enable(){
   TRAC_INSTANCE=${1:-$TRAC_INSTANCE} trac-configure components:tracwiki2sphinx.\*:enabled
}



#tracwiki2sphinx-branch(){    package-fn  $FUNCNAME $* ; }
#tracwiki2sphinx-basename(){  package-fn  $FUNCNAME $* ; }
#tracwiki2sphinx-dir(){       package-fn  $FUNCNAME $* ; }  
#tracwiki2sphinx-egg(){       package-fn  $FUNCNAME $* ; }
#tracwiki2sphinx-get(){       package-fn  $FUNCNAME $* ; }    

#tracwiki2sphinx-install(){   package-fn  $FUNCNAME $* ; }
#tracwiki2sphinx-uninstall(){ package-fn  $FUNCNAME $* ; } 
#tracwiki2sphinx-reinstall(){ package-fn  $FUNCNAME $* ; }
#tracwiki2sphinx-enable(){    package-fn  $FUNCNAME $* ; }  

#tracwiki2sphinx-status(){    package-fn  $FUNCNAME $* ; } 
#tracwiki2sphinx-auto(){      package-fn  $FUNCNAME $* ; } 
#tracwiki2sphinx-diff(){      package-fn  $FUNCNAME $* ; } 
#tracwiki2sphinx-rev(){       package-fn  $FUNCNAME $* ; } 
#tracwiki2sphinx-cd(){        package-fn  $FUNCNAME $* ; } 

#tracwiki2sphinx-fullname(){  package-fn  $FUNCNAME $* ; } 
#tracwiki2sphinx-update(){    package-fn  $FUNCNAME $* ; } 







