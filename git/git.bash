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


FUNCTIONS
----------

*git-bare*
       when invoked from the root of git working copy this
       creates a bare repo in *git-bare-dir* eg /var/scm/git 

*git-bare-scp name tag* 
       scp the bare git repo to remote node  


EOU
}
git-dir(){ echo $(local-scm-fold)/git ; }
git-cd(){  cd $(git-dir); }
git-mate(){ mate $(git-dir) ; }
git-make(){
   local dir=$(git-dir) &&  mkdir -p $dir 
}

git-bare-dir(){ echo $(git-dir) ; }
git-bare-path(){ echo $(git-bare-dir)/${1:-dummy}.git ; }
git-bare-name(){ echo ${GIT_BARE_NAME:-pycollada} ;}

git-bare(){
  local msg="=== $FUNCNAME :"
  echo $msg following recipe from http://www.git-scm.com/book/en/Git-on-the-Server-The-Protocols#The-HTTP/S-Protocol
  local path=$PWD
  [ ! -d "${path}/.git" ] && echo $msg needs to be invoked from toplevel of git checkout containing .git folder   && return 1
  local name=$(basename $path)
  local bare=$(git-bare-path $name)
  local hook=$bare/hooks/post-update
  [ -d "$bare" ] && echo $msg bare repo exists already at $bare && return 1
  local cmd="git clone --bare $path $bare ; mv $hook.sample $hook ; chmod a+x $hook  "
  echo $msg $cmd
  eval $cmd
}

git-bare-scp(){
  local msg="=== $FUNCNAME :"
  local name=${1:-$(git-bare-name)}
  local tag=${2:-N}
  [ "$NODE_TAG" == "$tag" ] && echo $msg cannot scp to self $tag && return 1 
  local cmd="scp -r $(git-bare-path $name) $tag:$(NODE_TAG=$tag git-bare-dir)"
  echo $msg $cmd
  eval $cmd
}

git-bare-clone(){
  local msg="=== $FUNCNAME :"
  local name=${1:-$(git-bare-name)}
  local bare=$(git-bare-path $name)
  [ ! -d "$bare" ] && echo $msg no bare git repo at $bare && return 1
  local cmd="git clone $bare"
  echo $msg $cmd
  eval $cmd 
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
