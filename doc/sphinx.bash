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


