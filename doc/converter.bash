# === func-gen- : doc/converter fgp doc/converter.bash fgn converter fgh doc
converter-src(){      echo doc/converter.bash ; }
converter-source(){   echo ${BASH_SOURCE:-$(env-home)/$(converter-src)} ; }
converter-vi(){       vi $(converter-source) ; }
converter-env(){      elocal- ; }
converter-usage(){ cat << EOU


Converter
==========

For converting latex into sphinx reStructured text (Georg Brandl)
as used by python project to migrate from latex to rst doc sources

  * http://svn.python.org/projects/doctools/converter/
  * http://svn.python.org/view/doctools/converter/

For trial docbuilding on non-nuwa capable nodes::

   scp -r N:/data1/env/local/dyb/NuWa-trunk/dybgaudi/Database/DybDbi/genDbi .
   (docs)simon:fig blyth$ scp N:/data1/env/local/dyb/NuWa-trunk/dybgaudi/Documentation/OfflineUserManual/fig/\*.png .

hookup to local apache with::

   apache-ln $PWD/_build/dirhtml oum
          
Installs
---------

Anon (readonly) installs onto WW:: 

    USER=anon converter-get


Usage 
------

converter-get
      checkout my fork from github
      On G, went into unexpected python: /Library/Python/2.5/site-packages/converter.egg-link 

converter-ln
      hookup to python

Pre-requisites
~~~~~~~~~~~~~~~

 * py25+  (py24 is NOT supported)
 * docutils

rst tips
~~~~~~~~~

 * finiky with indenting : if it dont work make sure lined up, and try adding some more space 

   == nuwa sphinx ==

      * nuwa (which includes py27)
      * virtualenv hookup 
         * {{{./dybinst trunk external virtualenv}}} 

      Install remainder into a virtual python env :

          * pip install sphinx

          * pip install -e git+git://github.com/simoncblyth/converter.git#egg=converter
                ## cannot push back to origin 
          * pip install -e git+git@github.com:simoncblyth/converter.git#egg=converter
                ## can push to origin, if have the key  
     
      Convert .tex sources to .rst
          * converter-test           

      Quickstart sphinx in the folder containing source .tex (and .rst)
          * sphinx-quickstart
          * make html 
          
      Configure nginx to serve the manual
          * http://belle7.nuu.edu.tw/oum

      Create pdf conversion with "make latexpdf", serve it at :
          * http://belle7.nuu.edu.tw/oum/OfflineUserManual.pdf
      via relative link from _build/html
          * OfflineUserManual.pdf -> ../latex/OfflineUserManual.pdf
 
      HMM : little point CMTifying this,  given its highly non-standard nature


    == linking to pdf without absolute URLs in source ? ==

       * want absolute pdf visible in the PDF document 
       * do not want base url in the sources

       * possiblilities 
            * sphinx global vars
            * dynamic conf.py 
            * sphinx-build -D OR -A options 
            * an absolutelink directive that uses a baseurl setting from conf ?

         make latexpdf SPHINXOPTS="-D latex_show_urls=True" > o
       
{{{
PDF render :download:`OfflineUserManual.pdf <_build/latex/OfflineUserManual.pdf>` available at URL `OfflineUserManual`_.

.. _OfflineUserManual: http://belle7.nuu.edu.tw/oum/_downloads/OfflineUserManual.pdf
.. target-notes::
}}}


    == math support ==

      Attempt to support math output in html by adding extension to conf.py 
      sphinx.ext.pngmath   

      Add to conf.py
          extensions = ['sphinx.ext.pngmath']

        * but the html output has latex errors embedded due to pkg inputenc missing  : "utf8x.def"

      sage:ticket:10350 would suggest that maybe able to live with utf8 rather than utf8x 
        * http://trac.sagemath.org/sage_trac/ticket/10350
        * http://www.mail-archive.com/sage-trac@googlegroups.com/msg53586.html
        
          pip install -U ipython
              ## ipython -pylab    may be doing smth underhand that requires ipython sibling to matplotlib

    == Git setup ==

         Description of setup moved to
              https://github.com/simoncblyth/converter/blob/master/NOTES


    == OTHERS ==

     Universal markup converter
         http://johnmacfarlane.net/pandoc/
         http://johnmacfarlane.net/pandoc/try

     See Imports on docutils links 
        http://docutils.sourceforge.net/docs/user/links.html
        http://docutils.sourceforge.net/rst.html 

   == ATTEMPT TO PARSE AND CONVERT DB SECTION OF OFFLINE USER MANUAL ==

    See how difficult conversion is the conversion of 
    realworld latex to reStructured text 
     
    For debugging conversions add to latexparser.parse_until  :
        print l,t,v,r  

    Changes to database_*.tex to parse and convert 

+            'center': '',
+            'tabular': '',
 
           \code         ... swapped to \tt
           \~/.my.cnf    ... \$HOME/.my.cnf 
           \begin{table} ... commented
            

    old python docs used some noddy approach to tables 
         http://docs.python.org/release/2.5.4/doc/table-markup.html
    which the converter supports ...
 


   = TODO =

    Handle the currently ignored commands :

em
end
normalsize
tt
caption
footnotesize
includegraphics
hbox

   == links ==

        http://docutils.sourceforge.net/docs/user/rst/quickref.html#hyperlink-targets

   == tables/tabular : DONE TO 0th ORDER  ==

         http://docutils.sourceforge.net/docs/ref/rst/directives.html#id25

         Use "table" directive in order to propagate a title, so can propagate table captions 
         like this  

   == figures/includegrapics ==

         \begin{figure}[ht]
         \begin{center}
         \includegraphics[scale=.35]{../fig/dbm_db_distribution}
         \caption{\label{dbm_db_distribution_fig}}
         \end{center}
         \end{figure}

         http://docutils.sourceforge.net/docs/ref/rst/directives.html#images

         .. figure:: path/to/file
            :scale: 50%

            Caption para goes here.





   == ATTEMPTING TO MAKE LATEX+PDF FROM THE RST ==

       C: RH4 latex too old utf8.def missing ... move to N



EOU
}
converter-dir(){ echo $(local-base)/env/doc/converter ; }
converter-cd(){  cd $(converter-dir)/$(converter-rdir) ; }
converter-mate(){ mate $(converter-dir) ; }

converter-url(){
   case $USER in 
      blyth) echo git@github.com:simoncblyth/converter.git ;;
          *) echo git://github.com/simoncblyth/converter.git  ;;
   esac 
}

converter-get(){
    local msg="=== $FUNCNAME :"
    local dir=$(dirname $(converter-dir)) &&  mkdir -p $dir && cd $dir
    [ -d converter ] && echo $msg converter dir exists already delete and rerun ... or use git && return 0
    
    local cmd="$SUDO ${PIP:-pip} install -e git+$(converter-url)#egg=converter"
    echo $msg $cmd
    eval $cmd 
}

converter-clone(){
    echo -n
}


converter-rdir(){ echo converter ; }

converter-texpath(){ echo Documentation/OfflineUserManual/tex ; }
converter-texdir(){ echo $DYB/NuWa-trunk/dybgaudi/$(converter-texpath) ; }
converter-rstdir(){
    local dir=/tmp/env/$FUNCNAME && mkdir -p $dir && echo $dir 
}

converter-sdir(){ echo $(env-home)/doc/converter ; }
converter-scd(){  cd $(converter-sdir) ; }

converter-ln-deprecated(){
    python-
    python-ln $(converter-dir)/converter
}
converter--(){   python $(converter-sdir)/convert.py ; }



converter-test(){
    [ -z "$DYB" ] && echo $msg DYB not defined && return 1
    cd $(converter-texdir)

    
}


converter-test-deprecated(){
    local msg="=== $FUNCNAME :"
    if [ -n "$DYB" ]; then 
       cd $(converter-texdir)
       echo $msg WARNING using the live docs in $PWD
    else 
        local dir=/tmp/env/$FUNCNAME && mkdir -p $dir && cd $dir
        [ ! -d "$(basename $(converter-texpath))" ] && svn co http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/$(converter-texpath)
        cd $(basename $(converter-texpath))
    fi

    local names="database_interface database_maintanence database_tables"
    local name
    for name in $names ; do
        echo converting $name.tex to $name.rst
        cat $name.tex | converter-- > $name.rst
    done 

}





