
invenio-env(){
  elocal-
}

invenio-source(){ echo $BASH_SOURCE ; }
invenio-usage(){

   cat << EOU 

      invenio-name : $(invenio-name)
      invenio-url  : $(invenio-url)

   


EOU

}

invenio-name(){     echo cds-invenio-0.99.1 ; }
invenio-basename(){ echo $(invenio-name).tar.gz ; }
invenio-url(){  echo http://cdsware.cern.ch/download/$(invenio-basename) ; }
invenio-dir(){  echo $LOCAL_BASE/env/invenio ; }
invenio-cd(){   cd $(invenio-dir) ; }

invenio-get(){
  mkdir -p $(invenio-dir)
  invenio-cd
  [ ! -f "$(invenio-basename)" ] && curl -O  $(invenio-url)
  [ ! -d "$(invenio-name)"     ] && tar zxvf "$(invenio-basename)" 

}


