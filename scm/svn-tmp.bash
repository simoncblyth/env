
svn-tmp-env(){
  
   elocal-

   heprez-
   apache-
}

svn-tmp-cp(){

    local srcs=/tmp/svn-apache2-conf-/etc/apache2/*
    for src in $srcs
    do
      local nam=$(basename $src)
      local tgt="$APACHE__LOCAL/$nam"
      local cmd="sudo cp $src $tgt && sudo chown $APACHE2_USER $tgt "
      echo $cmd 
    done


}



