svnsync-src(){    echo svn/svnsync/svnsync.bash ; }
svnsync-source(){ echo ${BASH_SOURCE:-$(env-home)/$(svnsync-src)} ; }
svnsync-vi(){     vi $(svnsync-source); }


svnsync-usage(){
   
   private-
   cat << EOU

svnsync 
=========

create env mirror on D
------------------------

svn to hg conversion testing requires lots of access to env repo. 
Doing that multiple times over the network is painfully slow, 
so try creating a local mirror.

Prepare destination : local mirror repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    fold=/var/scm/subversion
    mkdir -p $fold
    repo=$fold/env
    svnadmin create $repo

    echo '#!/bin/sh' > $repo/hooks/pre-revprop-change
    chmod +x $repo/hooks/pre-revprop-change


Setup properties for the sync
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tells the destination, where to mirror from::

    delta:~ blyth$ svnsync init file://$repo http://dayabay.phys.ntu.edu.tw/repos/env
    Copied properties for revision 0.

Kickoff a sync
~~~~~~~~~~~~~~~~

::


    delta:~ blyth$ svnsync sync file:///var/scm/subversion/env   # took less than 5 mins to mirror 4652 revisions
    # subsequent calls only transfer new revisions

    Can now checkout locally with::

    svn co file:///var/scm/subversion/env/trunk env  



old notes
----------


   IT SEEMS ...   
      this is not the way to do a quick sync from 
      the 0.10.4 mirror to my new 0.11 instance as have to 
      go all the way back to revision zero...
   
   BUT ...
      The 0.11 instance was created via a backup + restore of
      the 0.10.4 mirror so i think it already knows that it 
      is a mirror SO maybe I can just 
      "svnsync-syncronise" from C ?
      
         *  probably run into repository UUID problems ??  
         *  NOPE it JUST WORKED  

        [blyth@cms01 ~]$ svnsync-syncronise
        Committed revision 3786.
        Copied properties for revision 3786.
        Committed revision 3787.
        Copied properties for revision 3787.
        ...
        Copied properties for revision 3808.
        Committed revision 3809.
        Copied properties for revision 3809.
        Committed revision 3810.
        Copied properties for revision 3810.

   
        SVN_SYNC_USER : $(private-val SVN_SYNC_USER)
        NODE_TAG      : $NODE_TAG

    For simplicity the source/destination URLs are tied 
    to the invoking node with the invoking node being 
    the one to become the mirror

svnsync-sourceurl : $(svnsync-sourceurl)
svnsync-desturl   : $(svnsync-desturl)
svnsync-creds     : $(svnsync-creds)

svnsync-init-cmd  : $(svnsync-init-cmd)
               "desturl" "sourceurl" "creds"
     
svnsync-init 
    
         invoke the above command , which tells the destination repository
         which must be at revision zero that it is to be a mirror and the
         url of the source repository.

         THE DESTINATION REPOSITORY HAS TO BE AT REVISION ZERO
         AND SHOULD BE WRITABLE ONLY BY THE SVNSYNC USER ...   
            
         "svnsync help initialize" for details
            
svnsync-syncronize-cmd : $(svnsync-syncronize-cmd)
svnsync-syncronize
         invoke the above command, that tells the mirror "desturl" to update itself
         with respect to its "sourceurl"    

EOU

}


svnsync-env(){
   elocal-  
   
}





svnsync-sourceurl(){
  case ${1:-$NODE_TAG} in
    H) echo http://dayabay.ihep.ac.cn/svn/dybsvn ;;
    C) echo http://dayabay.ihep.ac.cn/svn/dybsvn ;;
    P) echo http://dayabay.ihep.ac.cn/svn/dybsvn ;;
    *) echo -n ;;
  esac
}
svnsync-desturl(){
  case ${1:-$NODE_TAG} in
    H) echo http://dayabay.phys.ntu.edu.tw/repos/mdybsvn ;;  
    C) echo http://cms01.phys.ntu.edu.tw/repos/mdybsvn ;;
    P) echo http://grid1.phys.ntu.edu.tw/repos/mdybsvn ;;
    *) echo -n ;;
  esac
}
svnsync-creds(){
  private-
  echo --username $(private-val SVN_SYNC_USER) --password $(private-val SVN_SYNC_PASS) 
}

svnsync-init-cmd(){
    echo svnsync initialize $(svnsync-desturl $*) $(svnsync-sourceurl $*) $(svnsync-creds $*)
}
svnsync-init(){
    local cmd=$(svnsync-init-cmd $*)
    echo $cmd
    eval $cmd
}

svnsync-syncronize-cmd(){
   echo svnsync synchronize $(svnsync-desturl $*) $(svnsync-creds $*)
}
svnsync-synchronize(){
    local cmd=$(svnsync-syncronize-cmd $*)
    echo $cmd
    eval $cmd  
}
