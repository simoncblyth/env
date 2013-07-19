# === func-gen- : scm/svn2git/svn2git fgp scm/svn2git/svn2git.bash fgn svn2git fgh scm/svn2git
svn2git-src(){      echo scm/svn2git/svn2git.bash ; }
svn2git-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svn2git-src)} ; }
svn2git-vi(){       vi $(svn2git-source) ; }
svn2git-env(){      elocal- ; }
svn2git-usage(){ cat << EOU

SVN2GIT via *git svn clone*
=============================

* http://git-scm.com/book/en/Git-and-Other-Systems-Migrating-to-Git


FUNCTIONS
-----------

*svn2git-clone name url*
      clone SVN repository into 

*svn2git-users*
      creates a users file that 
      assumes can treat all authors of all repos together

*svn2git-db-check*
      check authors

::

    simon:svn2git blyth$ echo select \* from authors \; | sqlite3 ~/.env/svnauthors.db 


issues
--------

slow
~~~~~

About 100 revisions per minute on G, so will take at least 40min for env


all users with any repo activity must be defined
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Otherwise the clone stops::

    r45 = 6b60eb834f145b3bc08713810729004ea8e71f9a (refs/remotes/trunk)
            A       db2trac/sources.xml
            A       db2trac/db2trac.xsl
            A       db2trac/test.sh
    Author: admin not defined in /usr/local/env/scm/svn2git/users.txt file


EOU
}

svn2git-dir(){ echo $(local-base)/env/scm/svn2git ; }
svn2git-users-path(){ echo $(svn2git-dir)/users.txt ; }
svn2git-db-path(){ echo  ~/.env/svnauthors.db ; }         # yuck, this is duplicating stuff in the config
svn2git-cd(){  cd $(svn2git-dir); }
svn2git-mate(){ mate $(svn2git-dir) ; }

svn2git-db-check(){
   echo select \* from authors \; | sqlite3 $(svn2git-db-path)
}

svn2git-users(){
   local users=$(svn2git-users-path)
   local db=$(svn2git-db-path)
   local script=$(env-home)/scm/svn2git/svnauthors.py 

   [   -f "$db" ] && echo $msg db file $db exists already delete and rerun to change
   [ ! -f "$db" ] && $script read && echo $msg populated DB

   [ "$db" -ot "$users" ] && echo db is older than users file $users nothing to do
   [ "$db" -nt "$users" ] && echo regenerate the users file $users && $script git > $users 

}

svn2git-clone(){
   local name=${1:-heprez}
   local repo=${2:-http://dayabay.phys.ntu.edu.tw/repos/$name/}
   local dir=$(svn2git-dir)
   [ ! -d "$dir" ] && mkdir -p $dir

   local users=$(svn2git-users-path)
   [ ! -f "$users" ] && echo $msg no users file $users : use svg2git-users to create this first && return 1

   local cmd="cd $dir && time git svn clone $repo --authors-file=$users --no-metadata --stdlayout $name"
   echo $msg $cmd
   eval $cmd
}

