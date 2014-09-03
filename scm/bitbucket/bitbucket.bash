# === func-gen- : scm/bitbucket/bitbucket fgp scm/bitbucket/bitbucket.bash fgn bitbucket fgh scm/bitbucket
bitbucket-src(){      echo scm/bitbucket/bitbucket.bash ; }
bitbucket-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bitbucket-src)} ; }
bitbucket-vi(){       vi $(bitbucket-source) ; }
bitbucket-env(){      elocal- ; }
bitbucket-usage(){ cat << EOU

BITBUCKET
==========

related
--------

* see *hg-* regarding the conversion from Subversion to Mercurial

Upload env Mercurial conversion to bitbucket
----------------------------------------------

#. create repo with bitbucket webinterface named "env", leave it private for now
#. adjust paths in the bare converted repo on D

::

    delta:~ blyth$ cd /var/scm/mercurial/env/.hg
    delta:.hg blyth$ vi hgrc     # this is a bare converted repo, so this is creating the file
    delta:.hg blyth$ cd ..
    delta:env blyth$ hg paths
    default = ssh://hg@bitbucket.org/simoncblyth/env

#. push up to bitbucket, took under 2 minutes

::

    delta:env blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/env
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 4621 changesets with 13955 changes to 4349 files


#. observations https://bitbucket.org/simoncblyth/env

   * cursory glance suggests all commits/history are there
     TODO: clone from bitbucket for systematic check 
 
   * blyth/lint/maqm : Author not mapped to a bitbucket user   
   * added and verified my gmail to use for this 

Trial Usage
------------

Use ssh not http, for auto authentication via key::

    delta:~ blyth$ mv env env.svn

    delta:~ blyth$ hg clone https://simoncblyth@bitbucket.org/simoncblyth/env
    http authorization required
    realm: Bitbucket.org HTTP
    user: simoncblyth
    password: interrupted!

    delta:~ blyth$ hg clone ssh://hg@bitbucket.org/simoncblyth/env
    destination directory: env
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    added 4636 changesets with 14033 changes to 4363 files
    updating to branch default
    3051 files updated, 0 files merged, 0 files removed, 0 files unresolved

Username Mapping
-------------------

https://confluence.atlassian.com/display/BITBUCKET/Set+your+username+for+Bitbucket+actions

Bitbucket requires that the email address you commit with matches a validated
email address on an account. On your local system, where you commit, both Git
and Mercurial allow you to configure a global username/email and a repository
specific username/email. If you do not specify a repository specific
username/email values, both systems use the global default. So, for example, if
your Bitbucket account has a validated email address of joe.foot@gmail.com, you
need to make sure your repository configuration is set for that username. Also,
make sure you have set your global username/email address configuration to a
validated email address.

Checking raw commits with bitbucket web interface shows no associated email address::

   # HG changeset patch
   # User blyth

*hg convert* has a usermap option, so can setup mapping from SVN users like
"blyth" to an email address verified with bitbucket. 


Bitbucket Teams
----------------

* https://confluence.atlassian.com/display/BITBUCKET/Teams+Frequently+Asked+Questions

Bitbucket Static pages 
-----------------------

* moved to *bitbucketstatic-*

Bitbucket Source Links
-----------------------

* https://bitbucket.org/simoncblyth/env/src/08b695ed6e2f5c959ec5f70486125dbd6272c1d9/cuda/cuda_launch.py?at=default

Source links including the hash correspond to a precise version, 
these are precise but otherwise are horribly long and look obnoxious.

Gleaned from rom notes in the below issue 

* https://bitbucket.org/site/master/issue/7150/option-to-replace-the-commit-hash-in-url

And by inspection find less precise but more readable src links 
including **tip** or **default** also work:

* https://bitbucket.org/simoncblyth/env/src/default/cuda/cuda_launch.py
* https://bitbucket.org/simoncblyth/env/src/tip/cuda/cuda_launch.py


EOU
}

bitbucket-repo(){  echo simoncblyth.bitbucket.org ; }
bitbucket-sdir(){ echo $(env-home)/scm/bitbucket/$(bitbucket-repo) ; }
bitbucket-scd(){  cd $(bitbucket-sdir); }
bitbucket-dir(){ echo $(bitbucket-htdocs) ; }
bitbucket-cd(){  cd $(bitbucket-dir); }
bitbucket-htdocs(){ echo $HOME/$(bitbucket-repo) ; }
bitbucket-export(){
   export BITBUCKET_HTDOCS=$(bitbucket-htdocs)
}

bitbucket-username(){ echo ${BITBUCKET_USERNAME:-simoncblyth} ; }
bitbucket-repo(){ echo /var/scm/mercurial/${1:-env} ; }

bitbucket-paths-(){ 
   local name=${1:-env}
   cat << EOC

[paths]
default = ssh://hg@bitbucket.org/$(bitbucket-username)/$name

EOC
}
bitbucket-paths(){
   local name=${1:-env}
   local repo=$(bitbucket-repo $name)
   local hgrc=$repo/.hg/hgrc

   [ ! -d "$(dirname $hgrc)" ] && echo $msg ERROR no .hg dir $(dirname $hgrc) && return
   [ ! -f "$hgrc" ] && echo $msg writing $hrgc && $FUNCNAME- $ name > $hgrc 
   [ -f "$hgrc" ] && echo $msg hgrc $hgcr && cat $hgrc 

}



