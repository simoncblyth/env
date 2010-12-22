# === func-gen- : doc/converter fgp doc/converter.bash fgn converter fgh doc
converter-src(){      echo doc/converter.bash ; }
converter-source(){   echo ${BASH_SOURCE:-$(env-home)/$(converter-src)} ; }
converter-vi(){       vi $(converter-source) ; }
converter-env(){      elocal- ; }
converter-usage(){
  cat << EOU
     converter-src : $(converter-src)
     converter-dir : $(converter-dir)


    For converting latex into sphinx reStructured text (Georg Brandl)
         http://svn.python.org/projects/doctools/converter/
         http://svn.python.org/view/doctools/converter/

           as use by python project to migrate from latex to rst doc sources


    == Usage ==

        converter-get
                checkout from SVN and patch 

        converter-ln
                 hookup to python

    == Pre-requisites ==

      * py25+  (py24 is NOT supported)
      * docutils


   == nuwa sphinx ==

      * nuwa (which includes py27)
      * virtualenv hookup 
         * {{{./dybinst trunk external virtualenv}}} 

      Install remainder into a virtual python env :

          * pip install sphinx

          * pip install -e git+git://github.com/scb-/converter.git#egg=converter
                ## cannot push back to origin 
          * pip install -e git+git@github.com:scb-/converter.git#egg=converter
                ## can push to origin, if have the key  
     
      Convert .tex sources to .rst
          * converter-test           
                ## TODO ... standalone python walker or main.tex "parser" for all 
                            OfflineUserManual, not just database dir  

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


         make latexpdf SPHINXOPTS="-D latex_show_urls=True" > o
       
{{{
PDF render :download:`OfflineUserManual.pdf <_build/latex/OfflineUserManual.pdf>` available at URL `OfflineUserManual`_.

.. _OfflineUserManual: http://belle7.nuu.edu.tw/oum/_downloads/OfflineUserManual.pdf
.. target-notes::
}}}



    == math support ==

      Attempt to support math output in html by adding extension to conf.py 
      sphinx.ext.pngmath   
        * but the html output has latex errors embedded due to pkg inputenc missing  : "utf8x.def"

      sage:ticket:10350 would suggest that may be able to live with utf8 rather than utf8x 
        * http://trac.sagemath.org/sage_trac/ticket/10350
        * http://www.mail-archive.com/sage-trac@googlegroups.com/msg53586.html

        

          pip install -U ipython
              ## ipython -pylab    may be doing smth underhand that requires ipython sibling to matplotlib



    == ISSUES ==

        On N, py24 doesnt do relative imports ..

   == Git setup ==

   === export from SVN into git repo ===

        cd $HOME
        svn info http://svn.python.org/projects/doctools/converter/ 

          Path: converter
          URL: http://svn.python.org/projects/doctools/converter
          Repository Root: http://svn.python.org/projects
          Repository UUID: 6015fed2-1504-0410-9fe1-9d1591cc4771
          Revision: 87407
          Node Kind: directory
          Last Changed Author: georg.brandl
          Last Changed Rev: 68972
          Last Changed Date: 2009-01-27 05:08:02 +0800 (Tue, 27 Jan 2009)

        svn export http://svn.python.org/projects/doctools/converter/ 
        
        cd converter
        git init
        git add README convert.py converter
         
        git commit -m "original latex to reStructuredText converter from python project, exported from  http://svn.python.org/projects/doctools/converter/ last changed rev 68972, revision 87407 "
        git remote add origin git@github.com:scb-/converter.git

            ## from github dashboard create a new repo named the same as this, ie "converter"

        git push origin master    
            ## if ssh-agent not running, you will be prompted for passphrase

        ## check the repo appears at   https://github.com/scb-/converter

        cd converter   ## apply the patch from the appropriate dir 
        patch -p0 < $(converter-patch-path)

        git add __init__.py docnodes.py latexparser.py restwriter.py tokenizer.py test_tokenizer.py 
        git commit -m "changes to support simple latex tabular environments in a more generic manner " 
                ## add the changes and commit

        git push origin master
                 ## push up to github  

         
         ## redirect the source handling in these functions to github and deprecate the patch manipulations 



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
      blyth) echo git@github.com:scb-/converter.git ;;
          *) echo git://github.com/scb-/converter.git  ;;
   esac 
}

converter-get(){
    local dir=$(dirname $(converter-dir)) &&  mkdir -p $dir && cd $dir
    [ -d converter ] && echo $msg converter dir exists already delete and rerun ... or use git && return 0
    #svn co http://svn.python.org/projects/doctools/converter/
    #converter-patch-apply
    git clone $(converter-url)
}

converter-rdir(){ echo converter ; }

converter-texpath(){ echo Documentation/OfflineUserManual/tex/database ; }
converter-texdir(){ echo $DYB/NuWa-trunk/dybgaudi/$(converter-texpath) ; }
converter-rstdir(){
    local dir=/tmp/env/$FUNCNAME && mkdir -p $dir && echo $dir 
}

converter-sdir(){ echo $(env-home)/doc/converter ; }
converter-scd(){  cd $(converter-sdir) ; }
converter-ln(){
    python-
    python-ln $(converter-dir)/converter
}

converter-dyb(){ python $(converter-sdir)/dyb.py ; }
converter--(){   python $(converter-sdir)/convert.py ; }

converter-test(){
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




converter-patch-path(){ echo $(converter-sdir)/converter-add-tabular.patch ; }
converter-deprecated-patch-apply(){
   [ ! -f "$(converter-patch-path)" ] && return 0
   converter-cd
   patch -p0 < $(converter-patch-path)
}
converter-deprecated-patch-make(){
    converter-cd
    svn diff > $(converter-patch-path) 
}

