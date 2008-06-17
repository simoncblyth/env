

bash-env(){
  elocal-
}

bash-usage(){

cat << EOU

  This just gives the filename ... not much use ... i want the directory 
      BASH_SOURCE  : $BASH_SOURCE

  the problem with bash functions is that they do not have a $0 to 
  tell them where they come from 
   ... problem arises as i like to provide a suite of functions to
  
  http://www.gnu.org/software/bash/manual/bashref.html#Invoking-Bash


EOU

}

bash-dir(){
   echo $(dirname $BASH_SOURCE)
}

bash-source(){
   echo $BASH_SOURCE
}


bash-funcdef(){
  local dir=$1
  local def="function func-dir(){ echo $dir ; }"
  echo $def
}


bash-create-func-with-a-func(){

   local def=$(bash-funcdef $(dirname $0))
   echo $def
   eval $def

}


bash-slash-count(){
   local s=$1
   local ifs=$IFS
   IFS=/
   bash-nargs $s 
   IFS=$ifs
   
}

bash-nargs(){
   echo $# 
}


