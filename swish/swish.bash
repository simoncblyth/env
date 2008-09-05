    
swishbuild-(){  . $ENV_HOME/swish/swishbuild/swishbuild.bash && swishbuild-env $* ; }

swish-usage(){

   cat << EOU
 
     swish-name   :  $(swish-name)
     swish-home   :  $(swish-home)
     which swish :  $(which swish)       
            
     swish-env :
                   invoked by precursor
                   sets up the PATH and LD_LIBRARY_PATH or ldconfig

     $(type swish-again)

     swish-test 
     
                   
     Precursors...
            
     swishbuild-


EOU

}



swish-notes(){

   cat << EOU
   
       Swish-e is a fast, flexible, and free open source system for indexing collections of Web pages or other files. 
   Swish-e is ideally suited for collections of a million documents or smaller. Using the GNOME? libxml2 parser and 
   a collection of filters, Swish-e can index plain text, e-mail, PDF, HTML, XML, Microsoft? Word/PowerPoint/Excel 
   and just about any file that can be converted to XML or HTML text. Swish-e is also often used to supplement databases 
   like the MySQL? DBMS for very fast full-text searching. Check out the full list of features.

   Swish-e was featured in the Linux Journal article How to Index Anything by Josh Rabinowitz.

   Swish-e is descended from the original web indexing program SWISH by WWW Hall of Famer Kevin Hughes.

EOU


}


sqlite-again(){

   swishbuild-
   swishbuild-again
   
}

swish-name(){
   echo swish-2.4.5
}

swish-home(){
   case ${1:-$NODE_TAG} in 
      H) echo $(local-base)/swish/$(swish-name) ;;
      *) echo $(local-system-base)/swish/$(swish-name) ;;
   esac
}

swish-env(){

   elocal-
   
   export SWISH_NAME=$(swish-name)
   export SWISH_HOME=$(swish-home)
   
   [ "$NODE_TAG" == "G" ] && return 0
   [ ! -d $SWISH_HOME ]  && return 0
   
   env-prepend $SWISH_HOME/bin

   case $NODE_TAG in
     P|XT|C) env-llp-prepend $SWISH_HOME/lib ;;
          *) env-ldconfig $SWISH_HOME/lib    ;;   ##  make available without diddling with llp
   esac     
}


swish-test(){
 echo -n
}





