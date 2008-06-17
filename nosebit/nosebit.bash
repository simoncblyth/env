
nosebit-env(){
   . $(nosebit-dir)/../python/pkg.bash
}

nosebit-dir(){
  echo $(dirname $BASH_SOURCE) 
}

nosebit-usage(){
cat << EOU

    nosebit-dir : $(nosebit-dir)
    
    Installation of packages needed for xml enhanced nosetests and
    bitten-slave automated test/build running.
      
      nosebit-install :
          does svn checkouts and installations into the site-packages of
          the python in the path, the invoking directory is used 
          as the working directory for checkouts 
          
      nosebit-install-test :
           install using temporary directory as working directory 
          
   TODO :
    
     * how to hookup nosebit to nuwa installation ?
    
     * where to put the nosebit working dir ? 
     
     * incorporate the parts of bitrun needed for slave running,
      
        - consider where to keep the slave cfg files 
        - what to put in them, how to auto create the .cfg files      
        - maybe something like :                   
        
          [nuwa]
          siteroot = /absolute/path/to/siteroot               
          release = trunk
          builds = False
          tests = True
             
                   
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



