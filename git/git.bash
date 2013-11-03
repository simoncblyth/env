# === func-gen- : git/git fgp git/git.bash fgn git fgh git
git-src(){      echo git/git.bash ; }
git-source(){   echo ${BASH_SOURCE:-$(env-home)/$(git-src)} ; }
git-vi(){       vi $(git-source) ; }
git-env(){      elocal- ; }
git-usage(){ cat << EOU

Git
====

Reference
------------

* http://gitref.org/remotes/#fetch
* http://book.git-scm.com/book.pdf 
* http://www.git-scm.com/book/en/Git-Basics-Getting-a-Git-Repository

Updating from remote branch
----------------------------

  git pull origin master    # 

Simple branching
------------------

* http://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging

::

    git checkout -b py25compat    # shorthand for "git branch" "git checkout" of the named branch  


Sharing a git repo
---------------------

* http://www.git-scm.com/book/en/Git-on-the-Server-The-Protocols#The-HTTP/S-Protocol




EOU
}
git-dir(){ echo $(local-base)/env/git/git-git ; }
git-cd(){  cd $(git-dir); }
git-mate(){ mate $(git-dir) ; }
git-get(){
   local dir=$(dirname $(git-dir)) &&  mkdir -p $dir && cd $dir

}


git-bare(){
  local msg="=== $FUNCNAME :"
  echo $msg following recipe from http://www.git-scm.com/book/en/Git-on-the-Server-The-Protocols#The-HTTP/S-Protocol
  local path=$1
  [ ! -f "${path}/.git" ] && echo $msg needs path to got repo argument && return 1
  local name=$(basename $path)
  apache-
  local hook=$name.git/hooks/post-update
  local cmd="cd $(apache-htdocs) ; git clone --bare $path $name.git ; mv $hook.sample $hook ; chmod a+x $hook/post-update "
  echo $cmd
}


git-conf(){

git config --global user.name "Simon Blyth"
git config --global user.email "blyth@hep1.phys.ntu.edu.tw"
git config --global color.diff auto
git config --global color.status auto
git config --global color.branch auto
git config --global core.editor "mate -w"

git config -l

}

git-learning(){

  local dir=/tmp/env/$FUNCNAME && mkdir -p $dir
  cd $dir


  git clone git://github.com/dcramer/django-compositepks.git
  git clone git://github.com/django/django.git


}
