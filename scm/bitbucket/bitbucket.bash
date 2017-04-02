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


IP Update
----------

::

    remote: We're changing our IP addresses on 15 December 2015 at 00:00 UTC.
    remote: Please make sure your firewalls are up to date:
    remote: https://blog.bitbucket.org/?p=2677



Commits showing up as DRAFT
-------------------------------

* https://bitbucket.org/site/master/issues/8678/draft-status-on-commits-bb-9791

The below removed all the DRAFT labels in web interface::

    cd ~/ImplicitMesher
    hg phase --public .
    hg push



Create Bitbucket repo from pristine zip/tarball 
---------------------------------------------------

::

    delta:isosurface blyth$ rm -rf ImplicitMesher   # remove the expanded distribition directory 
    delta:isosurface blyth$ 
    delta:isosurface blyth$ hg clone ssh://hg@bitbucket.org/simoncblyth/ImplicitMesher      # create an empty repo directory  
    destination directory: ImplicitMesher
    no changes found
    updating to branch default
    0 files updated, 0 files merged, 0 files removed, 0 files unresolved

    delta:isosurface blyth$ ll ImplicitMesher/
    drwxr-xr-x   6 blyth  staff  204 Apr  1 11:39 .hg

    delta:isosurface blyth$ unzip ImplicitMesher.zip     # unzip into the repo directory 
    Archive:  ImplicitMesher.zip
      inflating: ImplicitMesher/BlobSet.cpp  
      inflating: ImplicitMesher/BlobSet.h  
      inflating: ImplicitMesher/glut.h   
      ...

    delta:isosurface blyth$ cd ImplicitMesher   
    delta:ImplicitMesher blyth$ hg st         # status shows lots of untracked files
    ? BlobSet.cpp
    ? BlobSet.h
    ? ImplicitFunction.cpp
    ? ImplicitFunction.h
    ...

    ## add them, commit with message providing the distribution URL and push 

    delta:ImplicitMesher blyth$ hg commit -m "initial commit of pristine http://www.dgp.toronto.edu/~rms/software/ImplicitMesher/ImplicitMesher.zip "

    delta:ImplicitMesher blyth$ hg st .
    delta:ImplicitMesher blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/ImplicitMesher
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 1 changesets with 509 changes to 509 files
    delta:ImplicitMesher blyth$ 

    ## check appears on bitbucket : https://bitbucket.org/simoncblyth/implicitmesher/src

    ## clone into home using ssh, to allow modification via keys

    delta:~ blyth$ hg clone ssh://hg@bitbucket.org/simoncblyth/ImplicitMesher
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    destination directory: ImplicitMesher
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    added 1 changesets with 509 changes to 509 files
    updating to branch default
    509 files updated, 0 files merged, 0 files removed, 0 files unresolved
    delta:~ blyth$ 

    ## hack it into something usable.... , add README.rst and make public using "Settings" web interface

    delta:ImplicitMesher blyth$ hg commit -m "remove binaries, windows cruft and WildMagic "
    delta:ImplicitMesher blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/ImplicitMesher
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 1 changesets with 0 changes to 0 files

        



Upload Mercurial repo to bitbucket
----------------------------------------------

#. create repo in bitbucket webinterface named "env", "tracdev", etc..

   * NB old Safari I am using doesnt work with bitbucket web interface, 
     the Create Repository button is greay out... Chrome works

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
#. add a README.rst at top level in bitbucket dialect reStructuredText


Create empty opticksdata repo
------------------------------

* create in bitbucket web interface, "Repositories > Create New" naming it opticksdata
* clone the empty repo into home ::

    simon:~ blyth$ hg clone ssh://hg@bitbucket.org/simoncblyth/opticksdata
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    destination directory: opticksdata
    no changes found
    updating to branch default
    0 files updated, 0 files merged, 0 files removed, 0 files unresolved

    simon:~ blyth$ ll opticksdata/
    total 0
    drwxr-xr-x    6 blyth  staff   204 Jun  7 19:57 .hg
    drwxr-xr-x@ 137 blyth  staff  4658 Jun  7 19:57 ..
    drwxr-xr-x    3 blyth  staff   102 Jun  7 19:57 .

    simon:~ blyth$ cat opticksdata/.hg/hgrc 
    [paths]
    default = ssh://hg@bitbucket.org/simoncblyth/opticksdata




Draft Commits Showing in Bitbucket web interface
--------------------------------------------------

* https://bitbucket.org/site/master/issue/8678/draft-status-on-commits-bb-9791

Sept 19, 2014
~~~~~~~~~~~~~~~

* Observe commit from belle7 in web interface to be listed as in DRAFT status, possibly
  due to old client version on belle7

::

    [blyth@belle7 g4daeview]$ hg --version
    Mercurial Distributed SCM (version 1.5)

    delta:env blyth$ hg --version
    Mercurial Distributed SCM (version 2.8.1)



Overview README
----------------

* https://confluence.atlassian.com/display/BITBUCKET/Display+README+text+on+the+overview

Sept 17, 2014
~~~~~~~~~~~~~~~

Notice that README.rst is no longer being rendered as html for **env** 
but the other repos are ? 

* https://bitbucket.org/simoncblyth/env
* https://bitbucket.org/simoncblyth/tracdev
* https://bitbucket.org/simoncblyth/heprez
* https://bitbucket.org/simoncblyth/g4dae

Note that only env tries to use contents directive::

    delta:~ blyth$ grep contents  */README.rst 
    e/README.rst:.. contents:: :local:
    env/README.rst:.. contents:: :local:
    delta:~ blyth$ 



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

Sept 4th, 2014
~~~~~~~~~~~~~~~~~

#. Created "envsys" team https://bitbucket.org/envsys
#. Attempted to set permissions on https://bitbucket.org/simoncblyth/env to allow commits 


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


Setup Mercurial email address
------------------------------

**IMPORTANT** 

Before committing to a Mercurial repo make sure to 
set a ui/username in ~/.hgrc to a bitbucket validated 
email address::

    [ui]
    username = Simon Blyth <simoncblyth@gmail.com>


Bitbucket Unconfirmed email address
------------------------------------

An env mercurial commit from C2 prior to setting up 
email address resulted in a default of blyth@cms02 being
ascribed to the commit.  This causes problems as that 
is not a valid email.

Cannot use bitbucket aliases to workaround it as
cannot confirm that address.  Rather than rebuilding 
the repository to fix the email, maybe easier to 
make that email address valid ?

* http://www.techotopia.com/index.php/Configuring_an_RHEL_6_Postfix_Email_Server


  



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



