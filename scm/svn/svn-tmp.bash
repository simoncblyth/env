
svn-tmp-env(){
  
   elocal-

   ## nasty cross repo dependencies ...
   heprez-
   apache-
}

svn-tmp-cp(){

    local msg="=== $FUNCNAME :"
	echo $msg copy over the temporaries ... in order to get the repo accessible 

    local srcs=/tmp/svn-apache2-conf-/etc/apache2/*
    for src in $srcs
    do
      local nam=$(basename $src)
      local tgt="$APACHE__LOCAL/$nam"
      local cmd="sudo cp $src $tgt && sudo chown $APACHE2_USER $tgt "
      echo $cmd 
    done


}



