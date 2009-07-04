func-src(){      echo base/func.bash ; }
func-source(){   echo ${BASH_SOURCE:-$(env-home)/$(func-src)} ; }
func-vi(){       vi $(func-source) ; }
func-env(){      elocal- ; }
func-usage(){
  cat << EOU
     func-src : $(func-src)

EOU
}
func-end-template(){ echo 10 ; }
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
 
EON
}


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

func-gen-(){

  local msg="=== $FUNCNAME :"
  local fgp=$(func-gen-path $*)
  local fgn=$(func-gen-name $*)
  echo \# $msg $* fgp $fgp fgn $fgn

  head -$(func-end-template) $(func-source) | perl -p -e "s,$(func-src),$fgp," - | perl -p -e "s,func,$fgn,g" - 

}

func-precursor-(){
  local fgp=$(func-gen-path $*)
  local fgn=$(func-gen-name $*)
cat << EOP
$fgn-(){      . \$(env-home)/$fgp && $fgn-env \$* ; }
EOP
}



func-gen(){

  local msg="=== $FUNCNAME :"

  local arg=$1
  local repo=${2:-env}


  local fgp=$(func-gen-path $arg)
  local fgn=$(func-gen-name $arg)

  echo  $msg $arg $repo .... $fgp $fgn 


  local path=$($repo-home)/$fgp  
  local dir=$(dirname $path)
  local top=$($repo-home)/$repo.bash

  [ ! -f "$top" ] && echo $msg the repo $repo must have a top .bash at $top && return 1
  [ -f "$path" ]  && echo $msg ABORT : path $path exists already ... delete and rerun to override && return 0  
  
  echo;echo $msg proposes to write the below into : $path ;echo
  func-gen- $fgp 

  echo;echo $msg and hookup precursor into : $top;echo 
  func-precursor- $fgp
  echo

  local ans
  read -p "$msg enter YES to proceed : " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 0 
 
  [ ! -d "$dir" ] && echo $msg WARNING : creating dir $dir &&  mkdir -p "$dir" 
  func-gen- $fgp > $path
  func-precursor- $fgp >> $top


  eval $(func-precursor- $fgp)

  echo $msg $fgn-vi
  eval $fgn-vi

}


