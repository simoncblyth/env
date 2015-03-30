# === func-gen- : graphics/wendy/wendy fgp graphics/wendy/wendy.bash fgn wendy fgh graphics/wendy
wendy-src(){      echo graphics/wendy/wendy.bash ; }
wendy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(wendy-src)} ; }
wendy-vi(){       vi $(wendy-source) ; }
wendy-env(){      elocal- ; }
wendy-usage(){ cat << EOU


Wendy 
------

* https://github.com/elmindreda/Wendy

Wendy is a small and still incomplete game engine written 
by the author of GLFW, Camilla Berglund 


EOU
}
wendy-dir(){ echo $(local-base)/env/graphics/wendy ; }
wendy-cd(){  cd $(wendy-dir); }
wendy-mate(){ mate $(wendy-dir) ; }
wendy-get(){
   local dir=$(dirname $(wendy-dir)) &&  mkdir -p $dir && cd $dir

    git clone https://github.com/elmindreda/Wendy.git wendy 

}
