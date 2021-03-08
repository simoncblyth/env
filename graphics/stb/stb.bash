stb-source(){   echo ${BASH_SOURCE} ; }
stb-edir(){ echo $(dirname $(stb-source)) ; }
stb-ecd(){  cd $(stb-edir); }
stb-dir(){  echo $LOCAL_BASE/env/graphics/stb ; }
stb-cd(){   cd $(stb-dir); }
stb-vi(){   vi $(stb-source) ; }
stb-env(){  elocal- ; }
stb-usage(){ cat << EOU


https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/   



EOU
}
stb-get(){
   local dir=$(dirname $(stb-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d stb ] && git clone https://github.com/nothings/stb.git 

}
