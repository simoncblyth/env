# === func-gen- : tools/nasm fgp tools/nasm.bash fgn nasm fgh tools
nasm-src(){      echo tools/nasm.bash ; }
nasm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nasm-src)} ; }
nasm-vi(){       vi $(nasm-source) ; }
nasm-env(){      elocal- ; }
nasm-usage(){ cat << EOU

* http://www.nasm.us


EOU
}

nasm-ver(){ echo 2.13.01 ; }
nasm-nam(){ echo nasm-$(nasm-ver) ; }
nasm-url(){ echo http://www.nasm.us/pub/nasm/releasebuilds/$(nasm-ver)/nasm-$(nasm-ver).tar.gz ; }    
nasm-dir(){ echo $(local-base)/env/tools/nasm/$(nasm-nam) ; }
nasm-cd(){  cd $(nasm-dir); }


nasm-get(){
   local dir=$(dirname $(nasm-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(nasm-url) 
   local nam=$(nasm-nam) 
   local dst=$(basename $url)

   [ ! -f $dst ] && curl -L -O $url
   [ ! -d $nam ] && tar zxvf $dst
}

nasm-build(){

   nasm-cd
   ./configure --prefix=$(local-base)/env

   make          
   #make everything    # building docs fails on OSX for "cp -ufv" no -u option, and lack of some perl deps
   make install

}



