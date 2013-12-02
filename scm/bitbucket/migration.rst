Migration
==========

* https://confluence.atlassian.com/display/BITBUCKET/Import+code+from+an+existing+project
* http://blogs.atlassian.com/2011/03/goodbye_subversion_hello_mercurial/


git or mercurial
----------------

* https://confluence.atlassian.com/display/BITBUCKET/Set+up+Git+and+Mercurial

For Git, Bitbucket supports 1.6.6 or later and Mercurial version 1.7 or later

::

    simon:~ blyth$ git config --global --list
    user.name=Simon Blyth
    user.email=simon.c.blyth...
    color.diff=auto
    color.status=auto
    color.branch=auto

observations on github vs bitbucket
-------------------------------------

#. bitbucket supports password authentication whereas github requires ssh keys,
   this is distinctly easier albeit less secure.


meshlab into bitbucket
------------------------

Want to start from a pristine tarball for good housekeeping, and 
then progressively add in changes and commits for the so far unmanaged
changes made.  In order to move to standard (with SCM and frequent commits) 
development rather than the ad-hoc unmanaged approach.

Create new folder and explode the tarball into it::

    simon:graphics blyth$ mkdir mlb
    simon:graphics blyth$ cd mlb
    simon:mlb blyth$ tar xvf ../meshlab/MeshLabSrc_AllInc_v132.tar

Turn that into a git repo and add all the files::

    simon:mlb blyth$ git init 
    Initialized empty Git repository in /usr/local/env/graphics/mlb/.git/

Connect local git repo with the bitbucket one named to be **origin**::

    simon:mlb blyth$ git remote add origin https://scb-@bitbucket.org/scb-/meshlab.git

Add all the files to git staging area::

    simon:mlb blyth$ git status
    # On branch master
    #
    # Initial commit
    #
    # Untracked files:
    #   (use "git add <file>..." to include in what will be committed)
    #
    #       how_to_compile.txt
    #       meshlab/
    #       vcglib/
    nothing added to commit but untracked files present (use "git add" to track)
    simon:mlb blyth$ git add how_to_compile.txt meshlab vcglib 
    simon:mlb blyth$ git status
    # On branch master
    #
    # Initial commit
    #
    # Changes to be committed:
    #   (use "git rm --cached <file>..." to unstage)
    #
    #       new file:   how_to_compile.txt
    #       new file:   meshlab/src/README.txt
    #       new file:   meshlab/src/common/GLLogStream.cpp
    #       new file:   meshlab/src/common/GLLogStream.h
    ...
    #       new file:   vcglib/wrap/tsai/tsaimethods.cpp
    #       new file:   vcglib/wrap/tsai/tsaimethods.h
    #       new file:   vcglib/wrap/utils.h
    #
    simon:mlb blyth$   


Commit to local git with attribution in commit message::

    simon:mlb blyth$ git commit -m "Original MeshLab 1.3.2 obtained from sourceforge downloaded tarball MeshLabSrc_AllInc_v132.tar "
    [master (root-commit) 4f3aed5] Original MeshLab 1.3.2 obtained from sourceforge downloaded tarball MeshLabSrc_AllInc_v132.tar
     3991 files changed, 967722 insertions(+)
     create mode 100644 how_to_compile.txt
     create mode 100644 meshlab/src/README.txt
     create mode 100644 meshlab/src/common/GLLogStream.cpp
     create mode 100644 meshlab/src/common/GLLogStream.h
     ...

    simon:mlb blyth$ git status
    # On branch master
    nothing to commit, working directory clean
    simon:mlb blyth$ 

Push that to bitbucket and setup the bitbucket repo as upstream::

    simon:mlb blyth$ git push -u origin master
    Password for 'https://scb-@bitbucket.org': 
    Counting objects: 3702, done.
    Compressing objects: 100% (3620/3620), done.
    Writing objects: 100% (3702/3702), 11.94 MiB | 572 KiB/s, done.
    Total 3702 (delta 756), reused 0 (delta 0)
    Unable to rewind rpc post data - try increasing http.postBuffer
    error: RPC failed; result=65, HTTP code = 401
    fatal: The remote end hung up unexpectedly
    fatal: The remote end hung up unexpectedly
    Everything up-to-date
    simon:mlb blyth$ 

Checking web interface shows nothing there, so try again. Get the same error.

* https://bitbucket.org/scb-/meshlab/src
* http://blog.lukebennett.com/2011/07/25/git-broken-pipe-error-when-pushing-to-a-repository/
* https://bitbucket.org/site/master/issue/3578/cannot-push-fatal-the-remote-end-hung-up

Follow ticket suggestion to increase buffer succeeds to make the initial push::

    simon:mlb blyth$ git config --global http.postBuffer 524288000
    simon:mlb blyth$ git push -u origin master
    Password for 'https://scb-@bitbucket.org': 
    Counting objects: 3702, done.
    Compressing objects: 100% (3620/3620), done.
    Writing objects: 100% (3702/3702), 11.94 MiB | 1.51 MiB/s, done.
    Total 3702 (delta 756), reused 0 (delta 0)
    To https://scb-@bitbucket.org/scb-/meshlab.git
     * [new branch]      master -> master
    Branch master set up to track remote branch master from origin.
    simon:mlb blyth$ 

That works and now can see the initial commit in web interface::

* https://bitbucket.org/scb-/meshlab


Bitbucket Readme
-----------------

RST is supported.

* http://blog.bitbucket.org/2011/05/13/dress-up-your-repository-with-a-readme/
* https://bitbucket.org/stephenmcd/mezzanine/raw/473eb17e5a01770d8974d3ed7d66774cc0cc60e9/README.rst

Add README.rst

::

    simon:mlb blyth$ git status
    # On branch master
    # Untracked files:
    #   (use "git add <file>..." to include in what will be committed)
    #
    #       README.rst
    nothing added to commit but untracked files present (use "git add" to track)
    simon:mlb blyth$ 


::

    simon:mlb blyth$ git status            
    # On branch master
    # Your branch is ahead of 'origin/master' by 1 commit.
    #
    nothing to commit, working directory clean
    simon:mlb blyth$ 


Push::

    simon:mlb blyth$ git push origin master
    Password for 'https://scb-@bitbucket.org': 
    Counting objects: 4, done.
    Compressing objects: 100% (3/3), done.
    Writing objects: 100% (3/3), 777 bytes, done.
    Total 3 (delta 0), reused 0 (delta 0)
    To https://scb-@bitbucket.org/scb-/meshlab.git
       4f3aed5..7c40738  master -> master
    simon:mlb blyth$ 

Hmm its tedious entering password everytime need to setup ssh keys to avoid I guess.

* https://bitbucket.org/scb-/meshlab

Check cloning
--------------


Hmm, it works on G with git 1.8.0::

    simon:tmp blyth$ git --version
    git version 1.8.0

    simon:tmp blyth$  git clone https://scb-@bitbucket.org/scb-/meshlab.git
    Cloning into 'meshlab'...
    remote: Counting objects: 3705, done.
    remote: Compressing objects: 100% (2867/2867), done.
    remote: Total 3705 (delta 757), reused 3701 (delta 756)
    Receiving objects: 100% (3705/3705), 11.94 MiB | 1.77 MiB/s, done.
    Resolving deltas: 100% (757/757), done.
    Checking out files: 100% (3998/3998), done.

On N with git 1.5.5.6 it fails::

    [blyth@belle7 tmp]$ git --version
    git version 1.5.5.6
    [blyth@belle7 tmp]$ git clone https://scb-@bitbucket.org/scb-/meshlab.git
    Initialized empty Git repository in /tmp/meshlab/.git/
    Cannot get remote repository information.
    Perhaps git-update-server-info needs to be run there?
    [blyth@belle7 tmp]$ 

Hmm that git is too old

* Bitbucket supports from git 1.6.6 

Build git from src with `gitsrc-` and try again::

    [blyth@belle7 tmp]$ which git
    /data1/env/local/env/gitsrc/bin/git
    [blyth@belle7 tmp]$ git --version
    git version 1.8.5

Nope certificate issue::

    [blyth@belle7 tmp]$ git clone https://scb-@bitbucket.org/scb-/meshlab.git
    Cloning into 'meshlab'...
    fatal: unable to access 'https://scb-@bitbucket.org/scb-/meshlab.git/': SSL certificate problem, verify that the CA cert is OK. Details:
    error:14090086:SSL routines:SSL3_GET_SERVER_CERTIFICATE:certificate verify failed

    [blyth@belle7 tmp]$ git clone https://bitbucket.org/scb-/meshlab.git
    Cloning into 'meshlab'...
    fatal: unable to access 'https://bitbucket.org/scb-/meshlab.git/': SSL certificate problem, verify that the CA cert is OK. Details:
    error:14090086:SSL routines:SSL3_GET_SERVER_CERTIFICATE:certificate verify failed

    [blyth@belle7 tmp]$ git clone http://bitbucket.org/scb-/meshlab.git
    Cloning into 'meshlab'...
    fatal: unable to access 'https://bitbucket.org/scb-/meshlab.git/': SSL certificate problem, verify that the CA cert is OK. Details:
    error:14090086:SSL routines:SSL3_GET_SERVER_CERTIFICATE:certificate verify failed


Following

* http://stackoverflow.com/questions/3777075/ssl-certificate-rejected-trying-to-access-github-over-https-behind-firewall

disable SSL checking allows the clone, but its slow::

    [blyth@belle7 tmp]$ GIT_SSL_NO_VERIFY=true git clone http://bitbucket.org/scb-/meshlab.git
    Cloning into 'meshlab'...
    remote: Counting objects: 3705, done.
    remote: Compressing objects: 100% (2867/2867), done.




Git Remotes
----------------

* http://git-scm.com/book/en/Git-Basics-Working-with-Remotes

From ``git help push``::

  -u, --set-upstream
           For every branch that is up to date or successfully pushed, add
           upstream (tracking) reference, used by argument-less git-pull(1) and other
           commands. For more information, see branch.<name>.merge in git-config(1).




Do I always need to ``git push -u origin master`` ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``-u`` is needed once only to establish upstream tracking.

* http://longair.net/blog/2011/02/27/an-asymmetry-between-git-pull-and-git-push/






