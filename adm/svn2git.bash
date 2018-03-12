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


Timings for workflow : 5 min with ~1200 commits 
--------------------------------------------------

Considerable variability in timing, as depends on network.

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


Partial svn2git Conversion ?
-------------------------------

* https://daneomatic.com/2010/11/01/svn-to-multiple-git-repos/

Seems can easily convert a single svn folder into a git repo,
but not a selection of folders, unlike svn to mercurial.  
But can use gitfilter- to subsequently partition the fully 
converted git repo instead.



EOU
}
svn2git-repo(){ echo workflow ; }
svn2git-sdir(){ echo $HOME/$(svn2git-repo) ; }
svn2git-base(){ echo $(local-base)/env/adm/svn2git ; } 
svn2git-gdir(){ echo $(svn2git-base)/$(svn2git-repo) ; } 
svn2git-auth(){ echo $(svn2git-repo)-authors.txt ; }
svn2git-authpath(){ echo $(svn2git-base)/$(svn2git-auth) ; }

svn2git-svnurl(){ echo http://g4pb.local/repos/workflow ; }

svn2git-info(){ cat << EOI
$FUNCNAME
================

svn2git-base     : $(svn2git-base)
svn2git-sdir     : $(svn2git-sdir)
svn2git-gdir     : $(svn2git-gdir)
svn2git-authpath : $(svn2git-authpath)
svn2git-svnurl   : $(svn2git-svnurl)

Conversion of SVN repo, accessed via url into a full corresponding git repo 
with all the history revisions.

The git repo is written revision by revision into dir 
beneath svn2git-base : $(svn2git-base)

svn2git-clone-   : $(svn2git-clone-)

EOI
}

svn2git-scd(){   
    local sdir=$(svn2git-sdir)
    [ ! -d $sdir ] && echo expecting repo in sdir $sdir && return 
    cd $sdir
}
svn2git-cd(){  
    local base=$(svn2git-base)
    [ ! -d $base ] && mkdir -p $base
    cd $base
}

svn2git-authors-()
{
    svn log -q | awk -F '|' '/^r/ {sub("^ ", "", $2); sub(" $", "", $2); print $2" = "$2" <"$2">"}' | sort -u
}
svn2git-authors()
{
    local auth=$(svn2git-authpath)
    [ -f "$auth" ] && echo auth $auth exists already && return 

    local iwd=$PWD
    svn2git-scd
    $FUNCNAME- | cat > $(svn2git-authpath)
    cd $iwd
}
svn2git-authors-vi(){ vi $(svn2git-authpath)  ;}

svn2git-clone-(){ cat << EOC
git svn clone $(svn2git-svnurl) --prefix=origin/ --no-metadata -A $(svn2git-authpath) --stdlayout $(svn2git-repo) 
EOC
}
svn2git-clone(){
    local msg="=== $FUNCNAME :"
    svn2git-cd 
    local repo=$(svn2git-repo)
    [ -d "$repo" ] && echo $msg PWD $PWD repo $repo already converted : use svn2git-clone-wipe to scrub it  && return 

    $FUNCNAME-
    local cmd=$($FUNCNAME-)
    echo $cmd
    date 
    eval $cmd 
    date 
}

svn2git-clone-wipe(){
    local msg="=== $FUNCNAME :"
    svn2git-cd 
    local repo=$(svn2git-repo)
    [ ! -d "$repo" ] && echo $msg PWD $PWD repo $repo : no such directory to wipe && return 

    local cmd="rm -rf $repo"

    local ans
    read -p "$msg PWD $PWD : enter Y to proceed with [$cmd] : " ans

    [ "$ans" != "Y" ] && echo $msg OK skip && return 

    echo $msg proceed 
    eval $cmd 

}


svn2git--()
{   
   svn2git-clone-wipe
   svn2git-clone 
}

