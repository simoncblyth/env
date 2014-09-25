# === func-gen- : nuwa/dataquality fgp nuwa/dataquality.bash fgn dataquality fgh nuwa
dataquality-src(){      echo nuwa/dataquality.bash ; }
dataquality-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dataquality-src)} ; }
dataquality-vi(){       vi $(dataquality-source) ; }
dataquality-env(){      elocal- ; }
dataquality-usage(){ cat << EOU





EOU
}
dataquality-dir(){ echo $(local-base)/env/nuwa/DataQuality ; }
dataquality-cd(){  cd $(dataquality-dir); }
dataquality-mate(){ mate $(dataquality-dir) ; }
dataquality-get(){
   local dir=$(dirname $(dataquality-dir)) &&  mkdir -p $dir && cd $dir

   svn co http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/DataQuality

}
