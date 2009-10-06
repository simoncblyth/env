# === func-gen- : dj/djdev fgp dj/djdev.bash fgn djdev fgh dj
djdev-src(){      echo dj/djdev.bash ; }
djdev-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djdev-src)} ; }
djdev-vi(){       vi $(djdev-source) ; }
djdev-env(){      elocal- ; }
djdev-usage(){
  cat << EOU
     djdev-src : $(djdev-src)
     djdev-dir : $(djdev-dir)


     djdev-initialdata-path app   : $(dj-initialdata-path app)  
               
     djdev-dumpdata- app
             dump table serialization to stdout 
     
     djdev-dumpdata  app
             write table serialization to standard initialdata path

     djdev-loaddata  appname
             load table serialization into db
             CAUTION : this is done automatically on doing syncdb

    djdev-ln
          plant a symbolic link in site-package
          pointing at the version of django to use
          
          provides easy way to try out different versions ... simply change
          the dj-mode and rerun dj-ln to switch : no need to stop/start
          apache (when using MaxRequestsPerChild 1)
  

EOU
}
djdev-dir(){ echo $(local-base)/env/dj/dj-djdev ; }
djdev-cd(){  cd $(djdev-dir); }
djdev-mate(){ mate $(djdev-dir) ; }
djdev-get(){
   local dir=$(dirname $(djdev-dir)) &&  mkdir -p $dir && cd $dir

}

djdev-srcnam(){  
   case ${1:-$(djdev-mode)} in
    cpk) echo django-compositepks ;;
    pre) echo django$(djdev-cpkrev)   ;;
def|dev) echo django ;;
    svn) echo django ;;
    git) echo djgit  ;;
 system) echo django ;;
      *) echo django ;;
   esac 
}


djdev-cpk-mate(){ mate $(dj-srcfold)/$(dj-srcnam ${1:-cpk});  }

djdev-srcfold(){ 
   case $(djdev-mode $*) in 
      system) echo $(python-site) ;; 
           *) echo $(local-base)/env/django ;; 
   esac
}
djdev-mode(){ 
   case $NODE_TAG in 
     Z) echo system ;;
     G) echo dev ;;
     *) echo def ;;
   esac
}


djdev-git(){

   local dir=$(dj-srcfold)/djgit && mkdir -p $dir
   cd $dir
   [ ! -d .git ]  && git init 

   ## add some aliases to remotes  
   git remote add dcramer  git://github.com/dcramer/django-compositepks.git
   git remote add django   git://github.com/django/django.git

   ## access the urls
   git config --get remote.dcramer.url
   git config --get remote.django.url

   ## using the aliases for the fetch, stores the branches in eg django/master and dcramer/master 
   git fetch django
   git fetch dcramer
   
   ## create local branches to track the remote ones
   git checkout -b django  django/master 
   git checkout -b dcramer dcramer/master 

   ## list all branches , local and remote 
   git branch -a 


}


djdev-git-try(){

   local dir=$(dj-srcfold)
   cd $dir 
   [ ! -d "django-compositepks" ] && git clone git://github.com/dcramer/django-compositepks.git
   cd django-compositepks

   git branch aswas    ## for easy switchback

   [ "$(git config --get remote.django.url)" == "" ] && git remote add django git://github.com/django/django.git  

   
   git fetch django 

   echo $msg git merge django/master  ... and fix conflicts 

}


## cpk fork investigations ##

djdev-cpkurl(){  echo git://github.com/dcramer/django-compositepks.git ; }
djdev-cpkrev(){  echo 9477 ; }
djdev-cpk(){
    local msg="=== $FUNCNAME :"
    local dir=$(dj-srcfold)
    mkdir -p $dir && cd $dir  
   
    local cpk=$(dj-srcnam cpk)
    local pre=$(dj-srcnam pre)
    echo $msg clone/co $cpk and $pre and compare 
    [ ! -d "$cpk" ] && git clone $(dj-cpkurl)
    [ ! -d "$pre" ] && svn co    $(dj-srcurl)@$(dj-cpkrev) $pre

   
}


djdev-diff-(){
    local msg="=== $FUNCNAME :"
    local dir=$(dj-srcfold)
    cd $dir
    local aaa=$(dj-srcnam $1)
    local bbb=$(dj-srcnam $2)
    diff -r --brief $aaa $bbb | grep -v .svn | grep -v .pyc | grep Files  
}
djdev-diff(){
   $FUNCNAME- $* | while read line ; do
      dj-diff-parse $line $* || return 1
   done
}
djdev-diff-parse(){
   [ "$1" != "Files"  ] && return 1
   [ "$3" != "and"    ] && return 2
   [ "$5" != "differ" ] && return 3
   local aaa=$(dj-srcnam $6)/
   local bbb=$(dj-srcnam $7)/
   local a=${2/$aaa/}
   local b=${4/$bbb/}
   [ "$a" != "$b" ]  && return 4 
   echo $a   
}
djdev-opendiff(){
   local aaa=$(dj-srcnam $1)/
   local bbb=$(dj-srcnam $2)/
   djdev-diff $* | while read line ; do 
      echo opendiff $aaa$line $bbb$line 
   done
}


## fixtures : allow saving the state of tables ... for reloading on syncdb

djdev-initialdata-path(){ echo ${1:-theapp}/fixtures/initial_data.json ;  }
djdev-dumpdata-(){
    local app=${1:-dbi}
    local iwd=$PWD
    cd $(dj-projdir)  
    python manage.py dumpdata --format=json --indent 1 $*
    cd $iwd
}
djdev-dumpdata(){
    local msg="=== $FUNCNAME :"
    local app=${1:-dbi}
    local path=$(djdev-initialdata-path $app)
    local fixd=$(dirname $path)
    [ ! -d "$fixd" ] && echo $msg ABORT no dir $fixd && return 1
    [ -f "$path" ]   && echo $msg ABORT path $path already exists && return 2
    echo $msg app $app writing to $path 
    djdev-dumpdata- $app > $path
}
djdev-loaddata-(){
    local app=${1:-dbi}
    local iwd=$PWD
    cd $(dj-projdir)  
    python manage.py loaddata $*
    cd $iwd
}
djdev-loaddata(){
    local msg="=== $FUNCNAME :"
    local app=${1:-dbi}
    local path=$(djdev-initialdata-path $app)
    [ ! -f "$path" ]   && echo $msg ABORT no path $path && return 2
    echo $msg from $path 
    djdev-loaddata- $path
}



