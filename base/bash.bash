bash-vi(){ vi $BASH_SOURCE ; }

bash-env(){
  elocal-
}

bash-usage(){ cat << EOU

bash
=====

Redirect bug in bash 3.2
---------------------------

* https://stackoverflow.com/questions/1279953/how-to-execute-the-output-of-a-command-within-the-current-shell

::

   $ ls | sed ... | source /dev/stdin

UPDATE: This works in bash 4.0, as well as tcsh, and dash (if you change source to .). 
Apparently this was buggy in bash 3.2. From the bash 4.0 release notes::

    Fixed a bug that caused `.' to fail to read and execute commands 
    from non-regular files such as devices or named pipes.

Ref
----

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


bash-positional-args(){

  local msg="=== $FUNCNAME :"
  echo $msg initially $* 
  local args=$* 
  set -- 
  echo $msg after set $* 
  set $args 
  echo $msg after reset $* 


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



bash-getopts-wierdness(){


cat <<  EOW

In a fresh shell this works once only ....

simon:base blyth$ . bash.bash  
simon:base blyth$ bash-getopts -r -x red greed blu
OPTFIND
after opt parsing red greed blu
dummy=-x
rebuild=-r


simon:base blyth$ bash-getopts -r -x red greed blu
OPTFIND
after opt parsing red greed blu
dummy=
rebuild=

EOW

}


bash-heredoc(){

  cat << EOS

Backticks do get expanded 

  uname :  `uname`
  date  :  `date`

EOS

}


bash-heredoc-quoted(){

  cat << 'EOS'

Backticks do NOT get expanded when quote the end token

  uname :  `uname`
  date  :  `date`

EOS

}





bash-getopts(){

   # http://www.linux.com/articles/113836

   #
   #  The options must come first ...
   #       bash-getopts -r red green blue
   # 
   #  otherwise they are ignored 
   #        bash-getopts red green blue -r 
   #


   echo raw args \$@:$@  \$*:$* 


   local rebuild=""
   local dummy=""


   ## leading colon causes error messages not ro be skipped
   local o
   while getopts "rx" o ; do      
      case $o in
        r) rebuild="-r";;
        x) dummy="-x" ;;
      esac
   done



   echo OPTFIND $OPTFIND
   shift $((${OPTIND}-1))

   env | grep OPT

   echo after opt parsing   $@
   local
}




#echo BASH_SOURCE $BASH_SOURCE 
