# === func-gen- : doc/sphinx fgp doc/sphinx.bash fgn sphinx fgh doc
sphinx-src(){      echo doc/sphinx.bash ; }
sphinx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sphinx-src)} ; }
sphinx-vi(){       vi $(sphinx-source) ; }
sphinx-env(){      elocal- ; }
sphinx-usage(){
  cat << EOU
    
 == installs ==

   G   macports py25-sphinx   used for ~/heprez/docs  
   
        py25 in use with heprez for un-recalled compatibility reasons
        possibly jima:avg jython related 


   C2 (newone May 2012) [WHILE BEST TO NOT DO MUCH ON REPO SERVER MACHINE : DOCS WOULD BE HANDY]

        easy_install pip
	pip install sphinx 


  == reStructuredText ==

    http://wolfmanx.bitbucket.org/ws-docutils/README-inline-comments.html

        tagging and comments in rst


  == extension ideas ==

    A mobile compatible theme/builder would be useful ...
       * there is a html for chm option, but compiling M$ 
         chm on linux is problematic and demands chm reader on device 

 == readthedocs ==

    Interesting marriage of django + sphinx .
    Suspect that progress on mobile themes made in this proj 

       * http://readthedocs.org/ ... 
           * https://github.com/rtfd/readthedocs.org
           * http://ericholscher.com/blog/2011/jan/11/read-docs-updates/
           * http://twitter.com/readthedocs    
       
   Mar 1st 2011 tweet...
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





  == tryout sphinx ==

      1) from directory with .rst
             sphinx-quickstart    
                   answered defaults for almost all questions asked
                   creates conf.py + Makefile ... 

      2) add basenames of the rst to the toctree in the index.rst 
         created by quickstart  (same indentation and spacing is required) 

         .. toctree::
               :maxdepth: 2

               database_interface 
               database_maintanence

       3) make html
             open _build/html/index.html  
             file:///tmp/env/converter-test/database/_build/html/database_interface.html

       4) make latexpdf
 
       5) publish with nginx 

            cd `nginx-htdocs`
            sudo ln -s /tmp/out/_build/html sphinxdemo 
            nginx-edit
            nginx-stop  ## sv will auto re-start with new comnfig

            http://cms01.phys.ntu.edu.tw/sphinxdemo/database_interface.html#concepts


  == OSX install with macports py26 ==

     Install plucked Jinja2 and Pygments 

     Installing sphinx-build script to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin
     Installing sphinx-quickstart script to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin
     Installing sphinx-autogen script to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin
  
     Need to sphinx-path on OSX to put these in PATH 

  == about reStructured text  ==

    Online rst renderer
       http://www.tele3.cz/jbar/rest/rest.html

   Links in reStructured text 

  hello_ cruel_

.. _cruel: world
.. _hello: there




 == C1 : pip ==

 [blyth@cms01 docs]$ make
 sphinx-build -b dirhtml -d _build/doctrees   . _build/dirhtml
 Making output directory...
 Running Sphinx v1.0.5

 Extension error:
 Could not import extension sphinx.ext.mathjax (exception: No module named mathjax)
 make: *** [dirhtml] Error 1

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


       PROBABLY NEEDS A NEWER PYTHON ???



  == C2 : pip ==

   Hmm better not to do much on repository holding machine anyhow.


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




  == G : macports ==

    To make the Python 2.5 version of Sphinx the one that is run
    when you execute the commands without a version suffix, e.g. 'sphinx-build',
        run:
	        port select --set sphinx py25-sphinx

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




EOU
}
sphinx-dir(){ echo $(local-base)/env/doc/sphinx ; }
sphinx-cd(){  cd $(sphinx-dir); }
sphinx-mate(){ mate $(sphinx-dir) ; }
sphinx-get(){
   local dir=$(dirname $(sphinx-dir)) &&  mkdir -p $dir && cd $dir
   hg clone http://bitbucket.org/birkenfeld/sphinx 
}

sphinx-path(){
   if [ "$(uname)" == "Darwin" ]; then
      export PATH=/opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin:$PATH
   fi
}


sphinx-version(){   python -c "import sphinx as _ ; print _.__version__  " ; }
sphinx-pkgsource(){ python -c "import sphinx as _ ; print _.__file__  " ;   }

sphinx-test(){
   cd  /tmp/env/converter-test
   cd database

   

}


