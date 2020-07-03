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

Releases
----------

* https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release

* https://docs.github.com/en/rest/reference/repos#releases   ## REST API

The Releases API replaces the Downloads API.




:google:`using both bitbucket and github at the same time`
--------------------------------------------------------------
 
* http://blog.kevinlee.io/2013/03/11/git-push-to-pull-from-both-github-and-bitbucket/


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




EOU
}
github-get(){
   local dir=$(dirname $(github-dir)) &&  mkdir -p $dir && cd $dir

}
