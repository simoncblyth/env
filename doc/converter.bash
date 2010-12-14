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
            

    parsing of tabular/figure/center needs work .. added placeholders :

    470     def handle_tabular_env(self):
    471         return EmptyNode()
    472     def handle_figure_env(self):
    473         return EmptyNode()
    474     def handle_center_env(self):
    475         return EmptyNode()
    476 

    ignore commands 

em
end
normalsize
tt
caption
footnotesize
includegraphics
hbox


    Changes to write the rst  
          converter.restwriter.WriterError: no handler for tabular environment



EOU
}
converter-dir(){ echo $(local-base)/env/doc/converter ; }
converter-cd(){  cd $(converter-dir); }
converter-mate(){ mate $(converter-dir) ; }
converter-get(){
   local dir=$(dirname $(converter-dir)) &&  mkdir -p $dir && cd $dir
   svn co http://svn.python.org/projects/doctools/converter/
}

converter-texdir(){ echo $DYB/NuWa-trunk/dybgaudi/Documentation/OfflineUserManual/tex/database ; }
converter-rstdir(){
   local dir=/tmp/env/$FUNCNAME && mkdir -p $dir && echo $dir 
}
converter-texinputs-dir(){
   local dir=/tmp/env/converter/texinputs && mkdir -p $dir && echo $dir 
} 
converter-texinputs-setup(){
   cp $(converter-texdir)/*.tex $(converter-texinputs-dir)/
}
converter-convert(){
    python $(converter-dir)/convert.py $(dirname $(converter-texinputs-dir)) $(converter-rstdir)
}

converter-sdir(){ echo $(env-home)/doc/converter ; }
converter-scd(){  cd $(converter-sdir) ; }

converter-ln(){
   python-
   python-ln $(converter-dir)/converter
}
