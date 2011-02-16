# === func-gen- : doc/sphinx fgp doc/sphinx.bash fgn sphinx fgh doc
sphinx-src(){      echo doc/sphinx.bash ; }
sphinx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sphinx-src)} ; }
sphinx-vi(){       vi $(sphinx-source) ; }
sphinx-env(){      elocal- ; }
sphinx-usage(){
  cat << EOU
     sphinx-src : $(sphinx-src)
     sphinx-dir : $(sphinx-dir)


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


