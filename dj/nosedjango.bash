# === func-gen- : dj/nosedjango fgp dj/nosedjango.bash fgn nosedjango fgh dj
nosedjango-src(){      echo dj/nosedjango.bash ; }
nosedjango-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nosedjango-src)} ; }
nosedjango-vi(){       vi $(nosedjango-source) ; }
nosedjango-env(){      elocal- ; }
nosedjango-usage(){
  cat << EOU
     nosedjango-src : $(nosedjango-src)
     nosedjango-dir : $(nosedjango-dir)


     typical usage :
         nosetests -v --with-django

     from project or app directory, see nosedjango-examples for use of the test Client  


     Save typing the options by creating $HOME/.noserc containing :

[nosetests]
verbosity=3
with-django=1
     

EOU
}
nosedjango-dir(){ echo $(local-base)/env/django/nosedjango ; }
nosedjango-cd(){  cd $(nosedjango-dir); }
nosedjango-mate(){ mate $(nosedjango-dir) ; }
nosedjango-get(){
   local dir=$(dirname $(nosedjango-dir)) &&  mkdir -p $dir && cd $dir
    hg clone http://hg.assembla.com/nosedjango
}

nosedjango-install(){
   nosedjango-cd

   sudo python setup.py develop
   #make clean

   nosetests -p

}

nosedjango-examples(){

   nosedjango-cd ; cd examples/project   
   nosetests -v --with-django --with-doctest --doctest-tests 

}
