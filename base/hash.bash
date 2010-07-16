# === func-gen- : base/hash fgp base/hash.bash fgn hash fgh base
hash-src(){      echo base/hash.bash ; }
hash-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hash-src)} ; }
hash-vi(){       vi $(hash-source) ; }
hash-env(){      elocal- ; }
hash-usage(){
  cat << EOU
     hash-src : $(hash-src)
     hash-dir : $(hash-dir)


EOU
}
hash-dir(){ echo $(local-base)/env/base/base-hash ; }
hash-cd(){  cd $(hash-dir); }
hash-mate(){ mate $(hash-dir) ; }
hash-get(){
   local dir=$(dirname $(hash-dir)) &&  mkdir -p $dir && cd $dir

}

hash-names(){ echo red green blue cyan magenta yellow ; }
hash-exe-(){ cat << EOX
#!/bin/sh
echo \$0
EOX
}
hash-exe(){
  local dir=$1 
  hash-exe- > $dir/hashtest    
  chmod ugo+x $dir/hashtest
}

hash-notes(){ cat << EON

   hash-test
      create multiple temporary dirs containing the "hashtest" exe
      change the PATH to pick each exe in turn and check if 
      the correct path is returned by "which" and invoking the command..

      if find that the wrong path is obtained can try to fix by
      using  hash -d \$(which hashtest) to forget the path to hashtest
      that is being incorrectly cached

          HASH_FIX=forget hash-test


EON
}

hash-test(){
  local origpath=$PATH
  local tmp=/tmp/$USER/$FUNCNAME && mkdir -p $tmp
  local col
  for col in $(hash-names) ; do
     local dir=$tmp/$col
     mkdir -p $dir
     PATH=$dir:$PATH 
     hash-exe $dir

     case $HASH_FIX in 
       forget) hash -d $(which hashtest) ;;
     esac 

     local wht=$(dirname $(which hashtest))
     [ "$wht" == "$dir" ] && echo ok which $dir || echo mismatch which $dir $wht   

     local htr=$(hashtest)
     [ "$htr" == "$dir/hashtest" ] && echo ok invoke $htr || echo mismatch invoke $htr and $dir/hashtest   

  done
  PATH=$origpath
}



