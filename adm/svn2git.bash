# === func-gen- : adm/svn2git fgp adm/svn2git.bash fgn svn2git fgh adm
svn2git-src(){      echo adm/svn2git.bash ; }
svn2git-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svn2git-src)} ; }
svn2git-vi(){       vi $(svn2git-source) ; }
svn2git-env(){      elocal- ; }
svn2git-usage(){ cat << EOU

SVN2GIT
==========


Follow along the recipe:

* https://john.albin.net/git/convert-subversion-to-git
* https://github.com/JohnAlbin/git-svn-migrate

Retrieve a list of all Subversion committers
----------------------------------------------

::

    svn2git-authors-
    blyth = blyth <blyth>

    svn2git-authors- > ~/


Clone the Subversion repository using git-svn
-----------------------------------------------

* added the "--prefix=origin/" option cf the recipe 

::

    simon:~ blyth$ svn2git-clone
    WARNING: --prefix is not given, defaulting to empty prefix.
             This is probably not what you want! In order to stay compatible
             with regular remote-tracking refs, provide a prefix like
             --prefix=origin/ (remember the trailing slash), which will cause
             the SVN-tracking refs to be placed at refs/remotes/origin/*.
    NOTE: In Git v2.0, the default prefix will change from empty to 'origin/'.
    Initialized empty Git repository in /Users/blyth/workflow_git/.git/
    r1 = 871d3f97ab8e3aa311ea8bc7fe25a91fb43e8b3a (refs/remotes/trunk)
        A   workflow.bash
        A   drupal/drupalsql.sh


* http://stackoverflow.com/questions/31806817/how-to-choose-prefix-for-git-svn-clone
* https://bitbucket.org/atlassian/svn-migration-scripts/pull-requests/36

Saved terminal output to 

~/svn2git/svn2git-workflow.log
~/svn2git/svn2git-workflow-with-prefix.log


Repositioning inside svn2git
------------------------------

::

    simon:~ blyth$ l svn2git/
    total 1272
    -rw-r--r--@   1 blyth  staff  321554 May 11 14:38 svn2git-workflow-with-prefix.log
    drwxr-xr-x  137 blyth  staff    4658 May 11 14:38 workflow                 # 2nd try with prefix
    -rw-r--r--@   1 blyth  staff  321971 May 11 14:18 svn2git-workflow.log
    drwxr-xr-x  137 blyth  staff    4658 May 11 14:12 workflow_git             # 1st try without prefix
    -rw-r--r--    1 blyth  staff      48 May 11 14:03 authors-transform.txt





EOU
}
svn2git-dir(){ echo $(local-base)/env/adm/adm-svn2git ; }
svn2git-cd(){  cd $(svn2git-dir); }
svn2git-mate(){ mate $(svn2git-dir) ; }
svn2git-get(){
   local dir=$(dirname $(svn2git-dir)) &&  mkdir -p $dir && cd $dir

}

svn2git-workingcopy(){       echo ~/workflow ; }
svn2git-svnurl(){            echo http://g4pb.local/repos/workflow ; }
svn2git-gitdir(){            echo ~/svn2git/workflow ; }
svn2git-authors-transform(){ echo ~/svn2git/authors-transform.txt ; }

svn2git-authors-()
{
    cd $(svn2git-workingcopy)
    svn log -q | awk -F '|' '/^r/ {sub("^ ", "", $2); sub(" $", "", $2); print $2" = "$2" <"$2">"}' | sort -u
}
svn2git-authors()
{
    $FUNCNAME- | cat > $(svn2git-authors-transform) 
}
svn2git-authors-vi()
{
    vi $(svn2git-authors-transform) 
}
svn2git-clone(){
    cd 
    git svn clone $(svn2git-svnurl) --prefix=origin/ --no-metadata -A $(svn2git-authors-transform) --stdlayout $(svn2git-gitdir) 
}

