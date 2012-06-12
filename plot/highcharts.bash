# === func-gen- : plot/highcharts fgp plot/highcharts.bash fgn highcharts fgh plot
highcharts-src(){      echo plot/highcharts.bash ; }
highcharts-source(){   echo ${BASH_SOURCE:-$(env-home)/$(highcharts-src)} ; }
highcharts-vi(){       vi $(highcharts-source) ; }
highcharts-env(){      elocal- ; }
highcharts-usage(){ cat << EOU





EOU
}
highcharts-dir(){ echo $(local-base)/env/plot/$(highcharts-name) ; }
highcharts-cd(){  cd $(highcharts-dir); }
highcharts-mate(){ mate $(highcharts-dir) ; }
highcharts-name(){ echo Highcharts-2.2.5 ; }
highcharts-get(){
   local dir=$(dirname $(highcharts-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(highcharts-name)
   local zip=$nam.zip
   local url=http://www.highcharts.com/downloads/zips/$zip
   [ ! -f $zip ] && curl -L -O $url
   [ ! -d $nam ] && unzip $zip -d $nam

}
