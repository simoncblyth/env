# === func-gen- : tools/cuisine fgp tools/cuisine.bash fgn cuisine fgh tools
cuisine-src(){      echo tools/cuisine.bash ; }
cuisine-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cuisine-src)} ; }
cuisine-vi(){       vi $(cuisine-source) ; }
cuisine-env(){      elocal- ; }
cuisine-usage(){ cat << EOU

Cuisine
========

http://dayabay.phys.ntu.edu.tw/e/sysadmin/monitoring/

Installs
---------

G
~~
    ``sudo python setup.py install``   /opt/local/lib/python2.5/site-packages/cuisine-0.2.8-py2.5.egg

C2
~~~
     



EOU
}
cuisine-dir(){ echo $(local-base)/env/tools/cuisine ; }
cuisine-cd(){  cd $(cuisine-dir); }
cuisine-mate(){ mate $(cuisine-dir) ; }
cuisine-get(){
   local dir=$(dirname $(cuisine-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://github.com/sebastien/cuisine.git
}
