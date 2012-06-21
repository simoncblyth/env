# === func-gen- : nodejs/nodehighcharts fgp nodejs/nodehighcharts.bash fgn nodehighcharts fgh nodejs
nodehighcharts-src(){      echo nodejs/nodehighcharts.bash ; }
nodehighcharts-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nodehighcharts-src)} ; }
nodehighcharts-vi(){       vi $(nodehighcharts-source) ; }
nodehighcharts-env(){      elocal- ; }
nodehighcharts-usage(){ cat << EOU

https://github.com/davidpadbury/node-highcharts



EOU
}
nodehighcharts-dir(){ echo $(local-base)/env/nodejs/node-highcharts ; }
nodehighcharts-cd(){  cd $(nodehighcharts-dir); }
nodehighcharts-mate(){ mate $(nodehighcharts-dir) ; }
nodehighcharts-get(){
   local dir=$(dirname $(nodehighcharts-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://github.com/davidpadbury/node-highcharts.git

}
