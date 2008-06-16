
nosebit-env(){
   . $(nosebit-dir)/../python/pkg.bash
}

nosebit-dir(){
  echo $(dirname $BASH_SOURCE) 
}




nosebit-usage(){
cat << EOU

    nosebit-dir : $(nosebit-dir)
    
    Exploring how to distribute the packages needed for client testing ...
      nosebit-install :
          supplies the package name and urls to pkg-install which does the 
          checkouts and easy_install
          invoking directory is used as the working directory for checkouts 
          
      nosebit-install-test :
           install using temporary directory as working directory 
          
          
EOU
   pkg-usage
}

nosebit-install-test(){
    local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
    cd $tmp
   
    nosebit-install
}


nosebit-install(){ 
 pkg-install $(cat << EOT
     nose    http://python-nose.googlecode.com/svn/tags/0.10.3-release              524
   xmlnose   http://dayabay.phys.ntu.edu.tw/repos/env/trunk/xmlnose                 HEAD
   bitten    http://svn.edgewall.org/repos/bitten/branches/experimental/trac-0.11   547
EOT)
}

nosebit-entry-check(){
  pkg-entry-check easy_install nosetests bitten-slave
}



#nosebit-env
#nosebit-install



