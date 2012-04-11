# === func-gen- : db/dbxml fgp db/dbxml.bash fgn dbxml fgh db
dbxml-src(){      echo db/dbxml.bash ; }
dbxml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dbxml-src)} ; }
dbxml-vi(){       vi $(dbxml-source) ; }
dbxml-env(){   
   elocal- 
   bdbxml-
 }
dbxml-usage(){ cat << EOU

Usage of DBXML
~~~~~~~~~~~~~~

See also ``bdbxml-`` for a build and installation coverage.


DBXML-python Django mashup : SDPublisher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example of usage, rather than dependency to be used.

    http://www.sd-editions.com/SDPublisher/SDP110/SDPintro.html
    ( mkdir -p /tmp/sd ; cd /tmp/sd ; curl -O http://www.sd-editions.com/SDPublisher/SDP110/SDPublisher.zip && unzip SDPublisher.zip )

     ## messy exploding zip with loads .pyc and metadata __droppings 

      find . -name '*.pyc' -exec rm -f {} \;  
      find . -name '.DS_Store' -exec rm -f {} \;  

Interesting

#. XmlValue pythonic dressings




EOU
}
dbxml-dir(){ echo $(local-base)/env/db/db-dbxml ; }
dbxml-cd(){  cd $(dbxml-dir); }
dbxml-mate(){ mate $(dbxml-dir) ; }
dbxml-get(){
   local dir=$(dirname $(dbxml-dir)) &&  mkdir -p $dir && cd $dir

}

dbxml-py(){
   local pyc=$(python -c "import dbxml as _ ; print _.__file__ ")
   vi ${pyc/.pyc}.py
}
dbxml-cpp(){
   cd $BDBXML_HOME/include/dbxml
   pwd
}
