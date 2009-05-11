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


func-notes(){
   cat << EON

     func-gen- xml/xmldiff
          emit to stdout the filled out template for a new bash func   

     func-gen xml/xmldiff
          save the func-gen- output to $(env-home)/xml/xmldiff.bash 
          if no such file exists 

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
  local fgp=$(func-gen-path $*)
  local path=$(env-home)/$fgp  
  local dir=$(dirname $path)


  [ -f "$path" ]  && echo $msg ABORT : path $path exists already ... delete and rerun to override && return 0  
  [ ! -d "$dir" ] && echo $msg WARNING : creating dir $dir &&  mkdir -p "$dir" 
  echo $msg writing to path $path 

  func-gen- $fgp > $path
  cat $path

  echo $msg hook this up with precursor...
  func-precursor- $fgp 


}


