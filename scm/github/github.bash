# === func-gen- : scm/github/github fgp scm/github/github.bash fgn github fgh scm/github src base/func.bash
github-source(){   echo ${BASH_SOURCE} ; }
github-edir(){ echo $(dirname $(github-source)) ; }
github-ecd(){  cd $(github-edir); }
github-dir(){  echo $LOCAL_BASE/env/scm/github/github ; }
github-cd(){   cd $(github-dir); }
github-vi(){   vi $(github-source) ; }
github-env(){  elocal- ; }
github-usage(){ cat << EOU

Github
========

See Also
---------

* git-


forking opticks to opticks_ancient doesnt work 
-----------------------------------------------------

* https://stackoverflow.com/questions/72738008/github-cant-create-a-fork-from-a-branch-repository-already-exists

Well, I will answer my own question. Forking a repo with a different name
within the same org is not available yet, but it will be added very soon it
seems.

https://github.com/github/roadmap/issues/330



RSA key incident 24 March 2023
---------------------------------

* https://github.blog/2023-03-23-we-updated-our-rsa-ssh-host-key/

::

    epsilon:customgeant4 blyth$ git pull 
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
    Someone could be eavesdropping on you right now (man-in-the-middle attack)!
    It is also possible that a host key has just been changed.
    The fingerprint for the RSA key sent by the remote host is
    SHA256:uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s.
    Please contact your system administrator.
    Add correct host key in /Users/blyth/.ssh/known_hosts to get rid of this message.
    Offending RSA key in /Users/blyth/.ssh/known_hosts:15
    RSA host key for github.com has changed and you have requested strict checking.
    Host key verification failed.
    fatal: Could not read from remote repository.

    Please make sure you have the correct access rights
    and the repository exists.
    epsilon:customgeant4 blyth$ 
    epsilon:customgeant4 blyth$ 
    epsilon:customgeant4 blyth$ grep github.com ~/.ssh/known_hosts
    github.com,192.30.252.130 ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==
    epsilon:customgeant4 blyth$ 


    epsilon:customgeant4 blyth$ ssh-keygen -R github.com
    # Host github.com found: line 15
    /Users/blyth/.ssh/known_hosts:19: invalid line
    /Users/blyth/.ssh/known_hosts is not a valid known_hosts file.
    Not replacing existing known_hosts file because of errors
    epsilon:customgeant4 blyth$ 

After removing line 19::

    epsilon:customgeant4 blyth$ ssh-keygen -R github.com
    # Host github.com found: line 15
    /Users/blyth/.ssh/known_hosts updated.
    Original contents retained as /Users/blyth/.ssh/known_hosts.old
    epsilon:customgeant4 blyth$ 

    epsilon:customgeant4 blyth$ git pull
    Warning: the RSA host key for 'github.com' differs from the key for the IP address '140.82.121.3'
    Offending key for IP in /Users/blyth/.ssh/known_hosts:133
    Matching host key in /Users/blyth/.ssh/known_hosts:141
    Are you sure you want to continue connecting (yes/no)? yes
    Already up-to-date.
    epsilon:customgeant4 blyth$ 

Delete another offending line::

    epsilon:customgeant4 blyth$ vi ~/.ssh/known_hosts
    epsilon:customgeant4 blyth$ git pull
    Warning: Permanently added the RSA host key for IP address '140.82.121.3' to the list of known hosts.
    Already up-to-date.
    epsilon:customgeant4 blyth$ git pull
    Already up-to-date.
    epsilon:customgeant4 blyth$ 
    epsilon:customgeant4 blyth$ 




Github Pages
-------------

* https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages
* https://simoncblyth.github.io/index.html

* https://simoncblyth.github.io/


    epsilon:simoncblyth.bitbucket.io blyth$ pwd
    /Users/blyth/simoncblyth.bitbucket.io

    epsilon:simoncblyth.github.io blyth$ git remote -v
    origin	git@github.com:simoncblyth/simoncblyth.github.io.git (fetch)
    origin	git@github.com:simoncblyth/simoncblyth.github.io.git (push)

    epsilon:simoncblyth.bitbucket.io blyth$ git push github
    Counting objects: 98, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (98/98), done.
    Writing objects: 100% (98/98), 319.56 KiB | 2.85 MiB/s, done.
    Total 98 (delta 87), reused 0 (delta 0)
    remote: Resolving deltas: 100% (87/87), completed with 40 local objects.
    To github.com:simoncblyth/simoncblyth.github.io.git
       e64e253..b54dfb1  master -> master
    epsilon:simoncblyth.bitbucket.io blyth$ 


IHEP "Pages"
--------------

From workstation::

    ssh W
    cd simoncblyth.bitbucket.io
    git pull 

Laptop::

    open  https://juno.ihep.ac.cn/~blyth/



Opticks Github
-----------------

* https://github.com/simoncblyth/opticks
* https://github.com/simoncblyth/opticks/releases

Every few months : bring developments from bitbucket over to github
-----------------------------------------------------------------------

::

    cd ~/opticks
    git push github master  

    epsilon:opticks blyth$ git push github master 
    Counting objects: 3488, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (3472/3472), done.
    Writing objects: 100% (3488/3488), 985.84 KiB | 5.11 MiB/s, done.
    Total 3488 (delta 2917), reused 1 (delta 0)
    remote: Resolving deltas: 100% (2917/2917), completed with 481 local objects.
    To github.com:simoncblyth/opticks.git
       9a3d1888..fbdaba32  master -> master
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 


Every few months : add a git annotated tag, to enable JUNOEnv install and JUNO level testing
-----------------------------------------------------------------------------------------------



Nov 2020 tagging v0.1.0-rc1 creating downloadable github archive 
-----------------------------------------------------------------

::

    git tag -a v0.1.0-rc1 -m "4 months dev : GGeo/GNodeLib geometry rejig for universal transform access and triplet identifiers, SensorLib angular efficiency, watertight OCtx, G4Opticks getHit"  

    git push --tags           ## bitbucket is upstream
    git push github --tags    ## github is for infrequent pushes


    epsilon:opticks blyth$ git tag -a v0.1.0-rc1 -m "4 months dev : GGeo/GNodeLib geometry rejig for universal transform access and triplet identifiers, SensorLib angular efficiency, watertight OCtx, G4Opticks getHit" 
    epsilon:opticks blyth$ git tag
    v0.0.0-rc1
    v0.0.0-rc2
    v0.0.0-rc3
    v0.1.0-rc1
    epsilon:opticks blyth$ git push --tags 
    Counting objects: 1, done.
    Writing objects: 100% (1/1), 275 bytes | 275.00 KiB/s, done.
    Total 1 (delta 0), reused 0 (delta 0)
    To bitbucket.org:simoncblyth/opticks.git
     * [new tag]           v0.1.0-rc1 -> v0.1.0-rc1
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ git push github --tags
    Counting objects: 1, done.
    Writing objects: 100% (1/1), 275 bytes | 275.00 KiB/s, done.
    Total 1 (delta 0), reused 0 (delta 0)
    To github.com:simoncblyth/opticks.git
     * [new tag]           v0.1.0-rc1 -> v0.1.0-rc1
    epsilon:opticks blyth$ 



* https://github.com/simoncblyth/opticks/releases
* https://github.com/simoncblyth/opticks/archive/v0.1.0-rc1.tar.gz

* https://bitbucket.org/simoncblyth/opticks/commits/tag/v0.1.0-rc1

Check the tag is visible in both web interfaces.  On git the tag automatically 
gets downloadable archives (the primary motivation for the creating the tag). 


Releases
----------

* https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release

* https://docs.github.com/en/rest/reference/repos#releases   ## REST API

The Releases API replaces the Downloads API.


:google:`using both bitbucket and github at the same time`
--------------------------------------------------------------
 
* https://blog.kevinlee.io/blog/2013/03/11/git-push-to-pull-from-both-github-and-bitbucket/

pushing tags to github automatically makes archives
-----------------------------------------------------

See git-::

    git tag -a v0.0.0-rc1 -m "first test release"

    git push --tags           ## bitbucket is upstream
    git push github --tags    ## github is for infrequent pushes

::

    epsilon:tt blyth$ curl -L -O https://github.com/simoncblyth/opticks/archive/v0.0.0-rc1.tar.gz
    epsilon:tt blyth$ du -h v0.0.0-rc1.tar.gz
    5.1M	v0.0.0-rc1.tar.gz

    epsilon:tt blyth$ tar ztvf v0.0.0-rc1.tar.gz
    drwxrwxr-x  0 root   root        0 Jul  3 20:19 opticks-0.0.0-rc1/
    -rw-rw-r--  0 root   root       25 Jul  3 20:19 opticks-0.0.0-rc1/.gitignore
    -rw-rw-r--  0 root   root       83 Jul  3 20:19 opticks-0.0.0-rc1/.hgignore
    -rw-rw-r--  0 root   root     3073 Jul  3 20:19 opticks-0.0.0-rc1/CMakeLists.txt
    -rw-rw-r--  0 root   root     4253 Jul  3 20:19 opticks-0.0.0-rc1/CMakeLists.txt.old
    -rw-rw-r--  0 root   root    11416 Jul  3 20:19 opticks-0.0.0-rc1/LICENSE
    -rw-rw-r--  0 root   root      158 Jul  3 20:19 opticks-0.0.0-rc1/Makefile
    ...


Setup github as additional remote : for occasional pushing
----------------------------------------------------------------

::

    epsilon:opticks blyth$ git remote add github git@github.com:simoncblyth/opticks.git

    epsilon:opticks blyth$ git remote -v
    github	git@github.com:simoncblyth/opticks.git (fetch)
    github	git@github.com:simoncblyth/opticks.git (push)
    origin	git@bitbucket.org:simoncblyth/opticks.git (fetch)
    origin	git@bitbucket.org:simoncblyth/opticks.git (push)


For occasional use of github : ie when wanting to make a release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB not setting upstream, as that is hooked up to bitbucket::

    git push github master


For everyday use 
~~~~~~~~~~~~~~~~~~~

Normal advice for first push to a new github repo is::

    git push -u origin master    ## -u/--set-upstream
      
-u, --set-upstream
    For every branch that is up to date or successfully pushed, add
    upstream (tracking) reference, used by argument-less git-pull(1) and other
    commands. For more information, see branch.<name>.merge in git-config(1).

what is an upstream anyhow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/37770467/why-do-i-have-to-git-push-set-upstream-origin-branch

::

    epsilon:opticks blyth$ git --version
    git version 2.14.3 (Apple Git-98)

    epsilon:opticks blyth$ git config push.default
    simple


    git push github master 


"Syncing a Fork" : Updating my forked G4OpticksTest to get updates from Hans 
------------------------------------------------------------------------------

::

    epsilon:~ blyth$ git clone git@github.com:simoncblyth/G4OpticksTest.git
    Cloning into 'G4OpticksTest'...
    remote: Enumerating objects: 93, done.
    remote: Counting objects: 100% (93/93), done.
    remote: Compressing objects: 100% (69/69), done.
    remote: Total 333 (delta 50), reused 54 (delta 24), pack-reused 240
    Receiving objects: 100% (333/333), 165.44 KiB | 243.00 KiB/s, done.
    Resolving deltas: 100% (189/189), done.
    epsilon:~ blyth$ 
         
    epsilon:G4OpticksTest blyth$ git remote -v
    origin	git@github.com:simoncblyth/G4OpticksTest.git (fetch)
    origin	git@github.com:simoncblyth/G4OpticksTest.git (push)

    epsilon:G4OpticksTest blyth$ git remote add upstream https://github.com/hanswenzel/G4OpticksTest

    epsilon:G4OpticksTest blyth$ git remote -v
    origin	git@github.com:simoncblyth/G4OpticksTest.git (fetch)
    origin	git@github.com:simoncblyth/G4OpticksTest.git (push)
    upstream	https://github.com/hanswenzel/G4OpticksTest (fetch)
    upstream	https://github.com/hanswenzel/G4OpticksTest (push)

    epsilon:G4OpticksTest blyth$ git fetch upstream 
    remote: Enumerating objects: 298, done.
    remote: Counting objects: 100% (298/298), done.
    remote: Compressing objects: 100% (186/186), done.
    remote: Total 270 (delta 208), reused 134 (delta 80), pack-reused 0
    Receiving objects: 100% (270/270), 1.17 MiB | 208.00 KiB/s, done.
    Resolving deltas: 100% (208/208), completed with 24 local objects.
    From https://github.com/hanswenzel/G4OpticksTest
     * [new branch]      master     -> upstream/master
     * [new tag]         v0.1.1     -> v0.1.1
    epsilon:G4OpticksTest blyth$ 




EOU
}
github-get(){
   local dir=$(dirname $(github-dir)) &&  mkdir -p $dir && cd $dir

}
