# === func-gen- : doc/hieroglyph fgp doc/hieroglyph.bash fgn hieroglyph fgh doc
hieroglyph-src(){      echo doc/hieroglyph.bash ; }
hieroglyph-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hieroglyph-src)} ; }
hieroglyph-vi(){       vi $(hieroglyph-source) ; }
hieroglyph-env(){      elocal- ; }
hieroglyph-usage(){ cat << EOU

HIEROGLYPH : SPHINX SLIDE EXTENSION 
========================================

* http://yergler.net/blog/2012/03/13/hieroglyph/
* http://hieroglyph.io
* https://github.com/nyergler/hieroglyph

* http://code.google.com/p/html5slides/


Observations
-------------

Attempt to use autoslides False was unsuccessful, failed
to find slides deep into a tree.  In any case considering 
multiple presentations out of a single repo this is messy,
so try separate creation


USAGE
------

#. Create and cd to new directory for presentation::

    cd /work/blyth/presentations/2013/nov/fast_optical_photon_propagation_on_gpu

#. Run *hieroglyph-quickstart* 

   * other than title, author mostly defaults are fine
   * this creates: conf.py Makefile index.rst




CONFIG
-------

* http://docs.hieroglyph.io/en/latest/config.html#confval-slide_numbers
* http://docs.hieroglyph.io/en/latest/config.html#confval-autoslides

  * will need autoslides False, as would typically include 
    slide directives within a longer RST source


Docutils Issue in env
-----------------------

* :google:`docutils AssertionError: Losing "ids" attribute`

  * https://bitbucket.org/birkenfeld/sphinx/issue/1160/citation-target-missing-assertionerror


::

     18   File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/sphinx/environment.py", line 1504, in resolve_references
     19     builder.app.emit('doctree-resolved', doctree, fromdocname)
     20   File "/opt/local/lib/python2.5/site-packages/sphinx/application.py", line 314, in emit
     21     results.append(callback(self, *args))
     22   File "/opt/local/lib/python2.5/site-packages/hieroglyph-0.6.5.dev-py2.5.egg/hieroglyph/directives.py", line 193, in process_slideconf_nodes
     23     filter_doctree_for_slides(doctree)
     24   File "/opt/local/lib/python2.5/site-packages/hieroglyph-0.6.5.dev-py2.5.egg/hieroglyph/directives.py", line 171, in filter_doctree_for_slides
     25     child.traverse(no_autoslides_filter)
     26   File "/opt/local/lib/python2.5/site-packages/docutils/nodes.py", line 692, in replace_self
     27     'Losing "%s" attribute: %s' % (att, self[att])
     28 AssertionError: Losing "ids" attribute: ['todo']



Installs
----------

G
~~~

setup egg install
^^^^^^^^^^^^^^^^^^^

::

    simon:hieroglyph blyth$ sudo python2.5 setup.py install
    ## some py25 compatibility errors in the tests, but seems not the main package
    ...

    Adding hieroglyph 0.6.5.dev to easy-install.pth file
    Installing hieroglyph-quickstart script to /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin

    Best match: Sphinx 1.1.3
    Adding Sphinx 1.1.3 to easy-install.pth file
    Installing sphinx-apidoc script to /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin
    Installing sphinx-build script to /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin
    Installing sphinx-quickstart script to /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin
    Installing sphinx-autogen script to /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin

    Using /opt/local/lib/python2.5/site-packages
    Searching for docutils==0.8.1
    Best match: docutils 0.8.1
    Adding docutils 0.8.1 to easy-install.pth file

    Using /opt/local/lib/python2.5/site-packages
    Searching for Jinja2==2.6
    Best match: Jinja2 2.6
    Adding Jinja2 2.6 to easy-install.pth file

    Using /opt/local/lib/python2.5/site-packages
    Searching for Pygments==1.4
    Best match: Pygments 1.4
    Adding Pygments 1.4 to easy-install.pth file
    Installing pygmentize script to /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin

    Using /opt/local/lib/python2.5/site-packages
    Finished processing dependencies for hieroglyph==0.6.5.dev



simple link approach does not provide quickstart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sphinx-build is pinned to using macports py25::

    simon:e blyth$ head -1 `which sphinx-build`
    #!/opt/local/Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python

Switch to py25::

    sudo port select --list python  
    sudo port select --set python python25  

Check and add the link::

    simon:e blyth$ python-
    simon:e blyth$ python-site
    /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages

    simon:~ blyth$ hieroglyph-
    simon:~ blyth$ hieroglyph-ln
    simon:~ blyth$ which python  
    /opt/local/bin/python
    simon:~ blyth$ python -V
    Python 2.5.6
    simon:~ blyth$ python -c "import hieroglyph"

Back to py26::

    sudo port select --set python python26

    simon:~ blyth$ python-site
    /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages

Slides builder does not like AbtViz/docs/pyrootguithreads.rst probably due to orphan nature.
Hmm, is orphan my extension or is iy standard my sphinx is customized::

      1 
      2 :orphan: True
      3 
      4 Understanding PyROOT GUI Threads
      5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Builders
---------

* *slides*
* *dirslides*
* *inlineslides*
* *dirinlineslides*

Makefile integration
---------------------

::

    slides:
            $(SPHINXBUILD) -b slides $(ALLSPHINXOPTS) $(BUILDDIR)/slides
            @echo "Build finished. The HTML slides are in $(BUILDDIR)/slides."



SLIDE SHOW USAGE
------------------

*space bar*
           next slide
*left/right arrows* 
           move back/forward
*t*
           toggle table view
*c*
           presenters console to see whats coming next

RELATED
--------

* http://bitbucket.org/birkenfeld/sphinx-contrib
* https://pypi.python.org/pypi/sphinxcontrib-blockdiag/
* https://pypi.python.org/pypi/sphinxcontrib-googlemaps/
* http://sphinx-doc.org/ext/math.html
* https://pypi.python.org/pypi/tut/

  * tutorial style sphinx branch by branch

QUICKSTART
-----------

This runs before the normal Sphinx one::

    Welcome to the Hieroglyph 0.6.5.dev quickstart utility.

    This will ask questions for creating a Hieroglyph project, and then ask
    some basic Sphinx questions.


    The presentation title will be included on the title slide.
    > Presentation title: Testing Hieroglyph
    > Author name(s): Simon C Blyth

    Hieroglyph includes two themes:

    * slides
      The default theme, with different styling for first, second, and third
      level headings.

    * single-level
      All slides are styled the same, with the heading at the top.

    Which theme would you like to use?
    > Slide Theme [slides]: 


EOU
}
hieroglyph-dir(){ echo $(local-base)/env/doc/hieroglyph ; }
hieroglyph-cd(){  cd $(hieroglyph-dir); }
hieroglyph-mate(){ mate $(hieroglyph-dir) ; }
hieroglyph-get(){
   local dir=$(dirname $(hieroglyph-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(basename $(hieroglyph-dir)) 
   [ ! -d "$nam" ] && git clone https://github.com/nyergler/hieroglyph
}

hieroglyph-demodir(){ echo $(hieroglyph-dir).demo ; }
hieroglyph-demo-cd(){ 
   local dir=$(hieroglyph-demodir)
   mkdir -p $dir
   cd $dir
}
hieroglyph-envdir(){ echo $(hieroglyph-dir).env ; }
hieroglyph-env-cd(){ 
   local dir=$(hieroglyph-envdir)
   mkdir -p $dir
   cd $dir
}



hieroglyph-demo(){   
   $FUNCNAME-cd
   /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin/hieroglyph-quickstart 
}

hieroglyph-quickstart(){   
   $($FUNCNAME-path) $*
}
hieroglyph-quickstart-path(){   
   case $NODE_TAG in 
      G) echo /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/hieroglyph-quickstart ;;
   esac
}

hieroglyph-env-(){   
   ${FUNCNAME}cd
   cp $(env-home)/Makefile .
   cp $(env-home)/conf.py .
   cp -r $(env-home)/_templates .

}

hieroglyph-fix(){
   hieroglyph-cd
   perl -pi -e 's,child.traverse\(no_autoslides_filter\),child.traverse(no_autoslides_filter) or nodes.comment(),' src/hieroglyph/directives.py 
   git diff
}

hieroglyph-install(){
   local msg="=== $FUNCNAME :"
   hieroglyph-cd
   which python
   python -V
   python -c "import sys ; print '\n'.join(sys.path) " 
   echo 
   local ans
   read -p "$msg enter YES to proceed with installation into this python " ans 
   [ "$ans" != "YES" ] && echo $msg skipping && return 0 

   sudo python setup.py install
}
