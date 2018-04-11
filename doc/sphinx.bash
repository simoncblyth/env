# === func-gen- : doc/sphinx fgp doc/sphinx.bash fgn sphinx fgh doc
sphinx-src(){      echo doc/sphinx.bash ; }
sphinx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sphinx-src)} ; }
sphinx-vi(){       vi $(sphinx-source) ; }
sphinx-env(){      elocal- ; }
sphinx-usage(){
  cat << EOU

Sphinx
======


TODO
-----

* try to avoid the mods, via adding sphinxext  
* update to latest Sphinx, using github fork 


New Sphinx breakage of modified todo
--------------------------------------

::

    epsilon:home blyth$ find /Volumes/Delta/opt/local/Library/Frameworks/Python.framework -name todo.py

    /Volumes/Delta/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx/ext/todo.py

    diff /Volumes/Delta/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx/ext/todo.py sphinxext/todo.py

    path=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx/ext/todo.py

    vimdiff /Volumes/Delta$path $path




How to structure a large set of Sphinx docs ?
-----------------------------------------------

Python2
~~~~~~~~

* https://docs.python.org/2/contents.html
* https://docs.python.org/2/_sources/contents.txt


Python3
~~~~~~~~

* https://docs.python.org/3/contents.html
* https://docs.python.org/3/_sources/contents.txt


Salishsea
~~~~~~~~~~~~

Example:

* http://salishsea-meopar-docs.readthedocs.io/en/latest/code-notes/salishsea-nemo/index.html


Coding Guide
--------------

* http://www.sphinx-doc.org/en/stable/devguide.html#coding-guide

From the forked git clone::

   sphinx-cd

   git checkout master 


Create virtual environment python site::

    delta:~ blyth$ virtualenv /usr/local/env/doc/sphinx.env
    New python executable in /usr/local/env/doc/sphinx.env/bin/python
    Installing Setuptools..............................................................................................................................................................................................................................done.
    Installing Pip.....................................................................................................................................................................................................................................................................................................................................done.
    delta:~ blyth$ 


Set the envvars for the site::

    sphinx-cd   # check there the setup.py 

    . /usr/local/env/doc/sphinx.env/bin/activate  # get into environment

    pip install --editable .

    ## fails : presumably virtualenv/pip/setuptools installed from macports 
    ##         long ago is outdated 


::

    delta:sphinx blyth$ sphinx-cd
    WARNING THIS THIS NOT THE CURRENTLY USED SPHINX : use sphinx-scd instead for the macports sphinx
    delta:sphinx blyth$ 
    delta:sphinx blyth$ 
    delta:sphinx blyth$ . /usr/local/env/doc/sphinx.env/bin/activate 
    (sphinx.env)delta:sphinx blyth$ pip install --editable .
    Obtaining file:///usr/local/env/doc/sphinx
      Running setup.py egg_info for package from file:///usr/local/env/doc/sphinx
        error in Sphinx setup command: Invalid environment marker: python_version<"3.5"
        Complete output from command python setup.py egg_info:
        error in Sphinx setup command: Invalid environment marker: python_version<"3.5"

    ----------------------------------------
    Cleaning up...
    Command python setup.py egg_info failed with error code 1 in /usr/local/env/doc/sphinx
    Storing complete log in /Users/blyth/.pip/pip.log
    (sphinx.env)delta:sphinx blyth$ 




extlinks
-----------

* http://www.sphinx-doc.org/en/stable/ext/extlinks.html


quelling WARNING: nonlocal image URI found:
----------------------------------------------

::

    # warning suppresion that works in 1.2
    # https://stackoverflow.com/questions/12772927/specifying-an-online-image-in-sphinx-restructuredtext-format

    import sphinx.environment
    from docutils.utils import get_source_line

    def _warn_node(self, msg, node):
        if not msg.startswith('nonlocal image URI found:'):
            self._warnfunc(msg, '%s:%s' % get_source_line(node))

    sphinx.environment.BuildEnvironment.warn_node = _warn_node



CAUTION : Uncommitted mods in macports installed Sphinx
---------------------------------------------------------
::

    delta:sphinx blyth$ pwd
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx

    delta:sphinx blyth$ find . -name '*.py'  -exec grep -H SCB {} \;
    ...

::

    delta:sphinx blyth$ diff -y original_environment.py environment.py

    def process_images(self, docname, doctree):             def process_images(self, docname, doctree):
        """Process and rewrite image URIs."""                   """Process and rewrite image URIs."""
        for node in doctree.traverse(nodes.image):              for node in doctree.traverse(nodes.image):
            # Map the mimetype to the corresponding image.  T               # Map the mimetype to the corresponding image.  T
            # choose the best image from these candidates.  T               # choose the best image from these candidates.  T
            # set if there is only single candidate to be use               # set if there is only single candidate to be use
            # The special key ? is set for nonlocal URIs.               # The special key ? is set for nonlocal URIs.
            node['candidates'] = candidates = {}                    node['candidates'] = candidates = {}
            imguri = node['uri']                            imguri = node['uri']
            if imguri.find('://') != -1:                        if imguri.find('://') != -1:
                self.warn_node('nonlocal image URI found: %s'                   self.warn_node('nonlocal image URI found: %s'
                candidates['?'] = imguri                            candidates['?'] = imguri
                continue                                    continue
                                  >             elif imguri[0] == "@":       # SCB 
                                  >                 node['uri'] = imguri[1:]  # SCB
                                  >                 candidates['?'] = node['uri']  # SCB
                                  >                 self.warn_node('SCB DIRTY FIX asis image URI 
                                  >                 continue  # SCB
                                  >             else:   # SCB
                                  >                 pass   # SCB
                                  >             pass       # SCB
            rel_imgpath, full_imgpath = self.relfn2path(imgur               rel_imgpath, full_imgpath = self.relfn2path(imgur
            # set imgpath as default URI                        # set imgpath as default URI
            node['uri'] = rel_imgpath                           node['uri'] = rel_imgpath
            if rel_imgpath.endswith(os.extsep + '*'):                   if rel_imgpath.endswith(os.extsep + '*'):
                for filename in glob(full_imgpath):                     for filename in glob(full_imgpath):


Establish the actual diff
---------------------------

* actually no need, found original_environment.py 

::

    delta:sphinx blyth$ git checkout tags/1.2
    Note: checking out 'tags/1.2'.

    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by performing another checkout.

    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -b with the checkout command again. Example:

      git checkout -b new_branch_name

    HEAD is now at 2a86eff... Changelog bump.
    delta:sphinx blyth$ 



which one
-----------

* Every Sphinx page lists version at bottom right (currently 1.2)

::

    delta:wiki2rst blyth$ port installed 2>/dev/null | grep sphinx 
      py27-sphinx @1.2_0 (active)
      sphinx_select @0.1_0 (active)

    delta:wiki2rst blyth$ port contents py27-sphinx
    Port py27-sphinx contains:
      ...
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/Sphinx-1.2-py2.7.egg-info/PKG-INFO


version
--------

::

    delta:docxbuilder blyth$ which sphinx-build
    /opt/local/bin/sphinx-build

    delta:docxbuilder blyth$ sphinx-build --version
    Sphinx (sphinx-build) 1.2


extensions
-----------

* https://pypi.python.org/pypi/sphinxcontrib-newsfeed/0.1.1
* https://pypi.python.org/pypi/sphinxcontrib-googleanalytics/

List of sphinxcontrib extensions

* http://sphinxext-survey.readthedocs.org/en/latest/builders.html


update
--------

Macports sphinx is too old::

    py26-sphinx @1.1.3_1 (python, textproc, devel)
        Python documentation generator

::

    simon:doc blyth$ sphinx-cd .. 
    simon:doc blyth$ mv sphinx sphinx.dec2010
    simon:doc blyth$ sphinx-get


Careful where you check from::

    simon:sphinx blyth$ python -c "import sphinx ; print sphinx.__version__"
    1.2b3
    simon:sphinx blyth$ ( cd ; python -c "import sphinx ; print sphinx.__version__" )
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named sphinx

       
installs
---------

D : macports py27-sphinx (1.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Latex generated by Sphinx does not work with macports texlive-latex

#. File cmap.sty not found. fixed by macports texlive-latex-recommended

   * http://www.ctan.org/pkg/cmap  is used to make PDF files searchable

#. File titlesec.sty not found.  fixed by macports texlive-latex-extra


* https://trac.macports.org/wiki/TeXLivePackages  lists the content of the macports pkgs



N : system python 2.4
~~~~~~~~~~~~~~~~~~~~~~~~

Sphinx dependency issues, Pygments 1.5 has py24 incompatibility, downgrade to 1.4 with::

    sudo pip uninstall Pygments
    sudo pip install Pygments==1.4

Upgrade docutils from 0.6 to 0.9.1::

    sudo pip uninstall docutils
    sudo pip install docutils
G  
~~~

macports py25-sphinx   used for ~/heprez/docs  

      /opt/local/lib/python2.5/site-packages/sphinx      
   
py25 in use with heprez for un-recalled compatibility reasons
possibly jima:avg jython related 

Sphinx/Docutils Issue in attempt to use hieroglyph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Make a dirty fix, as dont want to try to update Sphinx just now::

    simon:hieroglyph.env blyth$ sudo vi /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/sphinx/environment.py

    1483                 elif typ == 'citation':
    1484                     docname, labelid = self.citations.get(target, ('', ''))
    1485                     if docname:
    1486                         newnode = make_refnode(builder, fromdocname, docname,
    1487                                                labelid, contnode)
    1488                     elif 'ids' in node:       ###  SCB DIRTY FIX  https://bitbucket.org/birkenfeld/sphinx/commits/72dceb35264e
    1489                         del node['ids'][:]    ###  SCB DIRTY FIX 
    1490                 # no new node found? try the missing-reference event
    1491                 if newnode is None:




C2 
~~~

(newone May 2012) [WHILE BEST TO NOT DO MUCH ON REPO SERVER MACHINE : DOCS WOULD BE HANDY]

easy_install pip
pip install sphinx 

WW 
~~~

pip installed v1.1.3, probably need newer python than 2.5.1 (but dont want to touch that on webserver)

::

       Exception occurred:
       File "/home/blyth/local/python/Python-2.5.1/lib/python2.5/site-packages/pygments/lexers/other.py", line 18, in <module>
             from pygments.lexers.web import HtmlLexer
          UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes in position 12-0: \Uxxxxxxxx out of range

reStructuredText
-------------------

http://wolfmanx.bitbucket.org/ws-docutils/README-inline-comments.html

        tagging and comments in rst


cross referencing 
------------------

**ref**
        tedious need to plant label

**doc**
        relative file name referencing, http://sphinx.pocoo.org/markup/inline.html#role-doc  from 0.6



extension ideas 
----------------

    A mobile compatible theme/builder would be useful ...
       * there is a html for chm option, but compiling M$ 
         chm on linux is problematic and demands chm reader on device 


customizing sphinx
-------------------

* :google:`sphinx RST customize`
* :google:`customize sphinx latex output`

* http://git-pull.readthedocs.org/en/latest/code_explorer/rst-docutils-sphinx-readthedocs.html
* https://github.com/github/markup/pull/220/files

readthedocs
-------------

Interesting marriage of django + sphinx .
Suspect that progress on mobile themes made in this proj 

 * http://readthedocs.org/ 

    * https://github.com/rtfd/readthedocs.org
    * http://ericholscher.com/blog/2011/jan/11/read-docs-updates/
    * http://twitter.com/readthedocs    
       
Mar 1st 2011 tweet

Pushed out the RTD Theme to everyone using the 'default' theme today. 
RTD Theme now includes sweet mobile styles using media queries.
 

  == advanced sphinx ==

     sphinx docs do not cover standard rst, although sphinx extends a lot of  
     of standard docutils so must be familiar with docutils to grok how
     Sphinx works

        http://docutils.sourceforge.net/docs/ref/rst/directives.html


  == SphinxReport : Using Sphinx for data access ==  

    http://wwwfgu.anat.ox.ac.uk/~andreas/SphinxReport/contents.html

     Uses ...
        * Python (2.5.2 or higher)
        * SQLAlchemy (0.4.8 or higher)
        * matplotlib (0.98.1 or higher)
        * sphinx (0.5-1 or higher)

      * adds an SQLAlchemy backend to Sphinx 

  == Features ==

    Pull docs out of python modules ..
         http://sphinx.pocoo.org/ext/autodoc.html#module-sphinx.ext.autodoc

    Lots of extensions : includes TracLinks 
         https://bitbucket.org/birkenfeld/sphinx-contrib/src

    Pretty html
         http://sphinx.pocoo.org/theming.html

    Tracker
         http://groups.google.com/group/sphinx-dev/topics

    reST renderer and demo 
         http://www.siafoo.net/tools/reST

    editor
        http://python.net/~gherman/ReSTedit.html


  == rst and Trac ==

    http://trac.edgewall.org/wiki/WikiRestructuredText

    trac browser obeys svn:mime-type property 

         svn propset svn:mime-type text/x-rst README.rst



  == demos ==

    http://people.ee.ethz.ch/~creller/web/tricks/reST.html
        includes demo of linebreaks inside table cells 


    http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html 
  
    http://sphinx.pocoo.org/markup/misc.html#tables
        points out reST tabularcolumns directive 
        hopefully can avoid htmlonly/latexonly bifurcation 


.. tabularcolumns:: |l|c|p{5cm}|

+--------------+---+-----------+
|  simple text | 2 | 3         |
+--------------+---+-----------+

    sphinx.ext.todo 
         make really red todo boxes



  == issues ==

Latex handles input encodings with '''inputenc'''
{{{
\usepackage[utf8]{inputenc}
}}}
   * the argument, typically '''utf8''' specifies the input encoding in usage
   * '''utf8x''' is non-standard encoding with lots of extra characters 

{{{
[blyth@belle7 ~]$ rpm -ql tetex-latex | grep utf8
/usr/share/texmf/tex/latex/base/utf8-test.tex
/usr/share/texmf/tex/latex/base/utf8.def
/usr/share/texmf/tex/latex/base/utf8enc.dfu
/usr/share/texmf/tex/latex/base/utf8test.tex
/usr/share/texmf/tex/latex/t2/test-utf8.tex
[blyth@belle7 ~]$
}}}

   * seems to be no yum pkg with utf8x.def

Hails from :
   * http://www.ctan.org/tex-archive/macros/latex/contrib/unicode/

Try test edit '''utf8x --> utf8''' in {{{/home/blyth/rst/lib/python2.7/site-packages/sphinx/ext/pngmath.py}}}
   * it works for simple math 


tryout sphinx 
===============

1.from directory with .rst
   
  :: 

     sphinx-quickstart    
         answered defaults for almost all questions asked
         creates conf.py + Makefile ... 

2. add basenames of the rst to the toctree in the index.rst 
   created by quickstart  (same indentation and spacing is required) 

   rst sources::

     .. toctree::
           :maxdepth: 2

           database_interface 
           database_maintanence

3. html::

         make html
         open _build/html/index.html  
         file:///tmp/env/converter-test/database/_build/html/database_interface.html

4. pdf:: 
    
         make latexpdf

5. publish with nginx:: 

        cd `nginx-htdocs`
        sudo ln -s /tmp/out/_build/html sphinxdemo 
        nginx-edit
        nginx-stop  ## sv will auto re-start with new comnfig

        http://cms01.phys.ntu.edu.tw/sphinxdemo/database_interface.html#concepts


OSX install with macports py26
=================================


Install plucked Jinja2 and Pygments::

     Installing sphinx-build script to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin
     Installing sphinx-quickstart script to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin
     Installing sphinx-autogen script to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin

Need to sphinx-path on OSX to put these in PATH 


 C1 : pip  : mathjax issue
=============================

PROBABLY NEEDS A NEWER PYTHON ???

::

     [blyth@cms01 docs]$ make
     sphinx-build -b dirhtml -d _build/doctrees   . _build/dirhtml
     Making output directory...
     Running Sphinx v1.0.5

     Extension error:
     Could not import extension sphinx.ext.mathjax (exception: No module named mathjax)
     make: *** [dirhtml] Error 1

::


     [blyth@cms01 docs]$ which sphinx-build
     /data/env/system/python/Python-2.5.1/bin/sphinx-build
     [blyth@cms01 docs]$ ll /data/env/system/python/Python-2.5.1/bin/sphinx-build
     -rwxr-xr-x  1 blyth blyth 300 Dec  2  2010 /data/env/system/python/Python-2.5.1/bin/sphinx-build
     [blyth@cms01 docs]$ 

     [blyth@cms01 docs]$ pip install --upgrade sphinx

      Running setup.py install for Pygments
          Sorry: UnicodeDecodeError: ('rawunicodeescape', '[\\U00010000-\\U0010FFFF]', 12, -1075636440, '\\Uxxxxxxxx out of range')
              Installing pygmentize script to /data/env/system/python/Python-2.5.1/bin
            Found existing installation: Sphinx 1.0.5
                Uninstalling Sphinx:


            [blyth@cms01 docs]$ make
            sphinx-build -b dirhtml -d _build/doctrees   . _build/dirhtml
            Running Sphinx v1.1.3

            Exception occurred:
              File "/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/pygments/lexers/other.py", line 18, in <module>
                  from pygments.lexers.web import HtmlLexer
                  UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes in position 12-0: \Uxxxxxxxx out of range
                  The full traceback has been saved in /tmp/sphinx-err-vfrOOA.log, if you want to report the issue to the developers.
                  Please also report this if it was a user error, so that a better error message can be provided next time.
                  Either send bugs to the mailing list at <http://groups.google.com/group/sphinx-dev/>,
                  or report them in the tracker at <http://bitbucket.org/birkenfeld/sphinx/issues/>. Thanks!
                  make: *** [dirhtml] Error 1
                  [blyth@cms01 docs]$ 





C2 : pip : unicode issue with old python
===========================================

Hmm better not to do much on repository holding machine anyhow.

::

    [blyth@cms02 ~]$ pip install sphinx
    Downloading/unpacking sphinx
      Downloading Sphinx-1.1.3.tar.gz (2.6Mb): 2.6Mb downloaded
        Running setup.py egg_info for package sphinx
	    /data/env/system/python/Python-2.5.1/lib/python2.5/distutils/dist.py:263: UserWarning: Unknown distribution option: 'use_2to3'
	          warnings.warn(msg)
		      /data/env/system/python/Python-2.5.1/lib/python2.5/distutils/dist.py:263: UserWarning: Unknown distribution option: 'use_2to3_fixers'
		            warnings.warn(msg)
			        no previously-included directories found matching 'doc/_build'

    warning: no previously-included files matching '*.pyo' found under directory 'docs'
      Running setup.py install for Pygments
          Sorry: UnicodeDecodeError: ('rawunicodeescape', '[\\U00010000-\\U0010FFFF]', 12, -1074987256, '\\Uxxxxxxxx out of range')
	      Installing pygmentize script to /data/env/system/python/Python-2.5.1/bin
	        Running setup.py install for sphinx
		    /data/env/system/python/Python-2.5.1/lib/python2.5/distutils/dist.py:263: UserWarning: Unknown distribution option: 'use_2to3'
		          warnings.warn(msg)
			      /data/env/system/python/Python-2.5.1/lib/python2.5/distutils/dist.py:263: UserWarning: Unknown distribution option: 'use_2to3_fixers'
			            warnings.warn(msg)
				        no previously-included directories found matching 'doc/_build'
					    Installing sphinx-apidoc script to /data/env/system/python/Python-2.5.1/bin
					        Installing sphinx-build script to /data/env/system/python/Python-2.5.1/bin
						    Installing sphinx-quickstart script to /data/env/system/python/Python-2.5.1/bin
						        Installing sphinx-autogen script to /data/env/system/python/Python-2.5.1/bin
							Successfully installed docutils Jinja2 Pygments sphinx


::

    [blyth@cms02 docs]$ make
    sphinx-build -b dirhtml -d _build/doctrees   . _build/dirhtml
    Making output directory...
    Running Sphinx v1.1.3

    Exception occurred:
    File "/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/pygments/lexers/other.py", line 18, in <module>
    from pygments.lexers.web import HtmlLexer
    UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes in position 12-0: \Uxxxxxxxx out of range
    The full traceback has been saved in /tmp/sphinx-err-pp6il1.log, if you want to report the issue to the developers.
    Please also report this if it was a user error, so that a better error message can be provided next time.
    Either send bugs to the mailing list at <http://groups.google.com/group/sphinx-dev/>,
    or report them in the tracker at <http://bitbucket.org/birkenfeld/sphinx/issues/>. Thanks!
    make: *** [dirhtml] Error 1
    [blyth@cms02 docs]$ 


G : macports 
--------------

To make the Python 2.5 version of Sphinx the one that is run
when you execute the commands without a version suffix, e.g. 'sphinx-build',
run::

	port select --set sphinx py25-sphinx

Find entry point "binaries"::

     g4pb-2:docs blyth$ port contents py25-sphinx | grep bin 
	/opt/local/bin/sphinx-apidoc-2.5
	/opt/local/bin/sphinx-autogen-2.5
	/opt/local/bin/sphinx-build-2.5
	/opt/local/bin/sphinx-quickstart-2.5

Problem with ``_md5``::

	g4pb-2:docs blyth$ sphinx-quickstart-2.5 
	Traceback (most recent call last):
	File "/opt/local/bin/sphinx-quickstart-2.5", line 9, in <module>
	load_entry_point('Sphinx==1.1.3', 'console_scripts', 'sphinx-quickstart')()
	File "/opt/local/lib/python2.5/site-packages/pkg_resources.py", line 337, in load_entry_point
		...
	from jinja2.loaders import BaseLoader, FileSystemLoader, PackageLoader, \
	File "/opt/local/lib/python2.5/site-packages/jinja2/loaders.py", line 19, in <module>
	from sha import new as sha1
	File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/sha.py", line 6, in <module>
	from hashlib import sha1 as sha
	File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/hashlib.py", line 133, in <module>
	md5 = __get_builtin_constructor('md5')
	File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/hashlib.py", line 60, in __get_builtin_constructor
	import _md5
	ImportError: No module named _md5

Resolved by installing the **py25-haslib** stub::

	   simon:~ blyth$ sudo port install py25-hashlib
	   --->  Computing dependencies for py25-hashlib
	   --->  Fetching archive for py25-hashlib
	   --->  Attempting to fetch py25-hashlib-2.5.4_1.darwin_9.ppc.tgz from http://packages.macports.org/py25-hashlib
	   ...
           --->  Activating py25-hashlib @2.5.4_1
	   --->  Cleaning py25-hashlib


G macports py26-sphinx
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:docs blyth$ sudo port install py26-sphinx
    Password:
    Warning: port definitions are more than two weeks old, consider using selfupdate
    --->  Computing dependencies for py26-sphinx
    --->  Dependencies to be installed: py26-docutils py26-jinja2 py26-markupsafe py26-pygments
    --->  Fetching archive for py26-docutils
    --->  Attempting to fetch py26-docutils-0.9.1_0.darwin_9.noarch.tbz2 from http://packages.macports.org/py26-docutils
    --->  Attempting to fetch py26-docutils-0.9.1_0.darwin_9.noarch.tbz2 from http://lil.fr.packages.macports.org/py26-docutils
    --->  Attempting to fetch py26-docutils-0.9.1_0.darwin_9.noarch.tbz2 from http://mse.uk.packages.macports.org/sites/packages.macports.org/py26-docutils
    --->  Fetching distfiles for py26-docutils
    --->  Attempting to fetch docutils-0.9.1.tar.gz from http://nchc.dl.sourceforge.net/project/docutils/docutils/0.9.1/
    --->  Verifying checksum(s) for py26-docutils
    --->  Extracting py26-docutils
    --->  Configuring py26-docutils
    --->  Building py26-docutils
    --->  Staging py26-docutils into destroot
    --->  Installing py26-docutils @0.9.1_0
    --->  Activating py26-docutils @0.9.1_0
    Error: org.macports.activate for port py26-docutils returned: Image error: /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/rst2xetex.py already exists and does not belong to a registered port.  Unable to activate port py26-docutils. Use 'port -f activate py26-docutils' to force the activation.
    Error: Failed to install py26-docutils
    Please see the log file for port py26-docutils for details:
        /opt/local/var/macports/logs/_opt_local_var_macports_sources_rsync.macports.org_release_ports_python_py-docutils/py26-docutils/main.log
    Error: The following dependencies were not installed: py26-docutils py26-jinja2 py26-markupsafe py26-pygments
    To report a bug, follow the instructions in the guide:
        http://guide.macports.org/#project.tickets
    Error: Processing of port py26-sphinx failed
    simon:docs blyth$ 

Had to force install it::

    simon:bin blyth$ sudo port install -f py26-docutils
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Computing dependencies for py26-docutils
    --->  Activating py26-docutils @0.11_0
    Warning: File /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/rst2xetex.py already exists.  Moving to: /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/rst2xetex.py.mp_1383719108.
    Warning: File /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/docutils/languages/lt.py already exists.  Moving to: /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/docutils/languages/lt.py.mp_1383719108.
    ...
    Warning: File /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/docutils/writers/xetex/__init__.pyc already exists.  Moving to: /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/docutils/writers/xetex/__init__.pyc.mp_1383719108.
    --->  Cleaning py26-docutils




EOU
}
sphinx-dir(){ echo $(local-base)/env/doc/sphinx ; }
sphinx-cd(){  
   echo $msg WARNING THIS THIS NOT THE CURRENTLY USED SPHINX : use sphinx-scd instead for the macports sphinx
   cd $(sphinx-dir)/$1; 
}


sphinx-sdir-old(){ 
  case $NODE_TAG in 
    G) echo /opt/local/lib/python2.5/site-packages/sphinx  ;;
    D) echo /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx ;; 
  esac
}

sphinx-sdir(){ python -c "import os, sphinx ; print os.path.dirname(sphinx.__file__) " ; }
sphinx-scd(){ cd ; cd $(sphinx-sdir) ; }   ## cd home to avoid getting confused by dirs called sphinx

sphinx-find()
{
   sphinx-scd
   pwd
   find . -name '*.py' -exec grep -H ${1:-sphinx-build} {} \;
}



sphinx-get(){
   local dir=$(dirname $(sphinx-dir)) &&  mkdir -p $dir && cd $dir
   #hg clone http://bitbucket.org/birkenfeld/sphinx   ## they migrated to github, where i forked it
   [ ! -d sphinx ] && git clone https://github.com/simoncblyth/sphinx
}





sphinx-install(){
   sphinx-cd
   which python
   python -V
   sudo python setup.py install 
}


sphinx-path(){
   if [ "$(uname)" == "Darwin" ]; then
      export PATH=/opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin:$PATH
   fi
}

sphinx-version(){   python -c "import sphinx as _ ; print _.__version__  " ; }
sphinx-pkgsource(){ python -c "import sphinx as _ ; print _.__file__  " ;   }
sphinx-build-speed(){
   local python=/opt/local/Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python 
   $python -c "import sphinx ; print sphinx.__version__"
   $python -c "import jinja2 ; print jinja2.__version__"
}


#sphinx-build(){
#   /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin
#}


sphinx-salishsea-note(){ cat << EON

This is an example of a large set of Sphinx docs 
structured and presented in a well thought out manner.

* https://salishsea-meopar-docs.readthedocs.io/en/latest/

* https://github.com/rtfd/sphinx_rtd_theme


EON
}

sphinx-salishsea-url(){ echo https://bitbucket.org/salishsea/docs ; }
sphinx-salishsea-dir(){ echo $LOCAL_BASE/env/doc/salishsea-docs ; }
sphinx-salishsea-cd(){  cd $(sphinx-salishsea-dir) ; }

sphinx-salishsea-get()
{
    local iwd=$PWD
    local dir=$(sphinx-salishsea-dir)
    local fold=$(dirname $dir)

    local url=$(sphinx-salishsea-url)
    local nam=$(basename $dir)

    [ ! -d $fold ] && mkdir -p $fold
    cd $fold
    [ ! -d $nam ] && hg clone $url $nam

    cd $iwd
}


