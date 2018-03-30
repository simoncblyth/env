func-src(){      echo base/func.bash ; }
func-source(){   echo ${BASH_SOURCE:-$(env-home)/$(func-src)} ; }
func-vi(){       vi $(func-source) ; }
func-env(){      elocal- ; }
func-usage(){ cat << EOU





EOU
}
func-dir(){ echo $(local-base)/env/heading/heading-func ; }
func-cd(){  cd $(func-dir); }
func-mate(){ mate $(func-dir) ; }
func-get(){
   local dir=$(dirname $(func-dir)) &&  mkdir -p $dir && cd $dir

}
func-end-template(){ echo 19 ; }
## CAUTION THE ABOVE IS A TEMPLATE FOR GENERATED FUNCS ... DO NOT EDIT

func-notes(){
   cat << EON

     func-gen- xml/xmldiff
          emit to stdout the filled out template for a new bash func   

     func-gen xml/xmldiff
          save the func-gen- output to $(env-home)/xml/xmldiff.bash 
          if no such file exists 

     func-gen base/hello <repo>
          generate $(<repo>-home)/base/hello/hello.bash 
          hook up the precursor into $(<repo>-home)/<repo>.bash
          eval the precursor


     func-isfunc-  name
          detect if the function "name" is defined 

          func-isfunc envx-
          n
          func-isfunc env-
          y
EON
}


func-isfunc-(){ local n=$1 ; [ "$(type $n 2>/dev/null | head -1 )" == "$n is a function" ] && return 0 || return 1 ; }
func-isfunc(){ $FUNCNAME- $* && echo y || echo n ;  }

func-gen-path(){
  local path=${1:-dummy/hello}
  local dir=$(dirname $path)
  local name=$(func-gen-name $path)
  echo $dir/$name.bash
}

func-gen-name(){
  local name=$(basename $1)  
  name=${name/.bash}
  echo $name  
}

func-gen-repo(){
  echo ${2:-env}
}

func-gen-heading(){
   local path=${1:-dummy/hello}
   local dir=$(dirname $path)
   echo $dir   
}


func-gen-(){

  local msg="=== $FUNCNAME :"
  local fgp=$(func-gen-path $*)
  local fgn=$(func-gen-name $*)
  local fgr=$(func-gen-repo $*)
  local fgh=$(func-gen-heading $*)
  local fgr0=${fgr:0:1}   ## first char of repo name

  echo \# $msg $* fgp $fgp fgn $fgn fgh $fgh

  head -$(func-end-template) $(func-source) \
         | perl -p -e "s,$(func-src),$fgp," - \
         | perl -p -e "s,func,$fgn,g" - \
         | perl -p -e "s,env-home,$fgr-home,g" - \
         | perl -p -e "s,/env,/$fgr,g" - \
         | perl -p -e "s,heading,$fgh,g" - \
         | perl -p -e "s,elocal-,${fgr0}local-,g" - \
         | cat 

}

func-precursor-(){
  local fgp=$(func-gen-path $*)
  local fgn=$(func-gen-name $*)
  local fgr=$(func-gen-repo $*)
cat << EOP
$fgn-(){      . \$($fgr-home)/$fgp && $fgn-env \$* ; }
EOP
}



func-gen(){

  local msg="=== $FUNCNAME :"

  local fgp=$(func-gen-path $*)
  local fgn=$(func-gen-name $*)
  local fgr=$(func-gen-repo $*)

  echo  $msg  .... fgp:$fgp fgn:$fgn fgr:$fgr

  local path=$($fgr-home)/$fgp  
  local dir=$(dirname $path)
  local top=$($fgr-home)/$fgr.bash

  [ ! -f "$top" ] && echo $msg the repo $repo must have a top .bash at $top && return 1
  [ -f "$path" ]  && echo $msg ABORT : path $path exists already ... delete and rerun to override && return 0  
  
  echo;echo $msg proposes to write the below into : $path ;echo
  func-gen- $*

  echo;echo $msg and hookup precursor into : $top;echo 
  func-precursor- $*
  echo

  local ans
  read -p "$msg enter YES to proceed : " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 0 
 
  [ ! -d "$dir" ] && echo $msg WARNING : creating dir $dir &&  mkdir -p "$dir" 
  func-gen- $* > $path
  func-precursor- $* >> $top


  echo $msg defining precursor $fgn- 
  eval $(func-precursor- $*)

  echo $msg invoking precursor $fgn-
  eval $fgn-
  eval $fgn-vi

}


