# === func-gen- : doc/sphinx fgp doc/sphinx.bash fgn sphinx fgh doc
sphinx-src(){      echo doc/sphinx.bash ; }
sphinx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sphinx-src)} ; }
sphinx-vi(){       vi $(sphinx-source) ; }
sphinx-env(){      elocal- ; }
sphinx-usage(){
  cat << EOU
     sphinx-src : $(sphinx-src)
     sphinx-dir : $(sphinx-dir)


    == tryout sphinx ==

       from the /tmp/out where converterd .rst are created
       
       setup the conf.py and other reqs : 
           sphinx-quickstart    ## used defaults for almost all questions asked

           make html

       cd `nginx-htdocs`
       sudo ln -s /tmp/out/_build/html sphinxdemo 
       nginx-edit
       nginx-stop  ## sv will auto re-start with new comnfig

    

       http://cms01.phys.ntu.edu.tw/sphinxdemo/database_interface.html#concepts



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

sphinx-version(){   python -c "import sphinx as _ ; print _.__version__  " ; }
sphinx-pkgsource(){ python -c "import sphinx as _ ; print _.__file__  " ;   }
