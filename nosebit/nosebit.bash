
nosebit-env(){
   elocal-
   pkg- 
}

nosebit-usage(){

  pkg-usage

cat << EOU

    BASH_SOURCE  : $BASH_SOURCE

     nosebit-install :
          supplies the package name and urls to pkg-install which does the 
          checkouts and easy_install

EOU

}


nosebit-install(){
 
 pkg-install $(cat << EOT
 
     nose    http://python-nose.googlecode.com/svn/tags/0.10.3-release              HEAD
   xmlnose   http://dayabay.phys.ntu.edu.tw/repos/env/trunk/xmlnose                 HEAD
   bitten    http://svn.edgewall.org/repos/bitten/branches/experimental/trac-0.11   HEAD

EOT)

}






