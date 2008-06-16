

pypi-usage(){
 
  cat << EOU

     http://pypi.python.org/pypi/pypicache/0.2
     http://pypi.python.org/pypi/EggBasket
                 uses genshi/turbogears to provide webapp to manage a package index


   pypicache ... does not do what i want, it is designed to set up an index for easy_install -i 
   egg distribution via apache mod_rewrite rules..  however its templates can be reused
    
   i want to create a local file index ... to be hosted in a repository that just  contains source
   url links    


      pypi-get :
           download and install ... provides inplace script and indexpackages module


      pypi-urls <names...> :
            generate simple python dict with package urls 
            eg:  pypi-urls bitten bitextra xmlnose nosenose
      

  
EOU


}



pypi-urls(){
  trac-
  nose-
  
  echo "{"
  for name in $*
  do
     $name-
     $name-cd 2> /dev/null && echo \'$name\':\"$(pypi-url)\",
  done
  echo "}"

}


pypi-url(){
   svn info | perl -n -e 'm/^URL: (\S*)$/ && print $1'
   
   
   
}


pypi-dir(){
  echo $LOCAL_BASE/env/python/pypi
}

pypi-cd(){
  cd $(pypi-dir)
}


pypi-get(){

  local dir=$(pypi-dir) && mkdir -p $dir
  [ ! -d $dir/pypicache ] && easy_install --editable -b $dir \
     --find-links http://pypi.python.org/packages/source/p/pypicache/pypicache-0.2.tar.gz#md5=706aacd3e224dc670eb561b5751f38d7 \
        pypicache
        
  cd $dir
  sudo easy_install -Z .


}



pypi-env(){
  echo -n
  elocal-
  python-
}


pypi-test(){

  cd $PYTHON_SITE  
  python /indexpackages.py $* . 

  

}

