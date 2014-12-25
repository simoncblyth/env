# === func-gen- : tools/graphviz/fdp fgp tools/graphviz/fdp.bash fgn fdp fgh tools/graphviz
fdp-src(){      echo tools/graphviz/fdp.bash ; }
fdp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fdp-src)} ; }
fdp-vi(){       vi $(fdp-source) ; }
fdp-env(){      elocal- ; }
fdp-usage(){ cat << EOU


   fdp -Tsvg g.dot -o g.svg 
   fdp -Tpng g.dot -o g.png 


EOU
}
fdp-dir(){ echo $(local-base)/env/tools/graphviz/tools/graphviz-fdp ; }
fdp-cd(){  cd $(fdp-dir); }
fdp-mate(){ mate $(fdp-dir) ; }
fdp-get(){
   local dir=$(dirname $(fdp-dir)) &&  mkdir -p $dir && cd $dir

}


fdp--(){
   local path=$1
   local name=$(basename $path)
   local base=${name/.dot}
   echo $path $name $base
   local cmd="fdp -Tpng $path -o $base.png"
   echo $cmd

}



