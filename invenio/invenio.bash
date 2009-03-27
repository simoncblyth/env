invenio-src(){    echo invenio/invenio.bash ; }
invenio-source(){ echo ${BASH_SOURCE:-$(env-home)/$(invenio-src)} ; }
invenio-vi(){     vi $(invenio-source) ; }
invenio-env(){
  elocal-
}

invenio-usage(){

   cat << EOU 


     http://cdsware.cern.ch/invenio/index.html
     http://cdsware.cern.ch/download/INSTALL


      invenio-name : $(invenio-name)
      invenio-url  : $(invenio-url)







  The prefork MPM provides a non-threaded, pre-forking 
  implementation that handles requests in a manner similar to Apache 1.3. 
  It is not as fast as threaded models, but is considered to be more stable. 
  It is appropriate for sites that need to maintain compatibility 
  with non-thread-safe libraries, and is the best MPM for isolating each request, 
  so that a problem with a single request will not affect any other. 


   


EOU

}

invenio-name(){     echo cds-invenio-0.99.1 ; }
invenio-basename(){ echo $(invenio-name).tar.gz ; }
invenio-url(){  echo http://cdsware.cern.ch/download/$(invenio-basename) ; }
invenio-dir(){  echo $(local-base)/env/invenio ; }
invenio-cd(){   cd $(invenio-dir) ; }

invenio-get(){
  mkdir -p $(invenio-dir)
  invenio-cd
  [ ! -f "$(invenio-basename)" ] && curl -O  $(invenio-url)
  [ ! -d "$(invenio-name)"     ] && tar zxvf "$(invenio-basename)" 

}


