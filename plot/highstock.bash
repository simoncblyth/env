# === func-gen- : plot/highstock fgp plot/highstock.bash fgn highstock fgh plot
highstock-src(){      echo plot/highstock.bash ; }
highstock-source(){   echo ${BASH_SOURCE:-$(env-home)/$(highstock-src)} ; }
highstock-vi(){       vi $(highstock-source) ; }
highstock-env(){      elocal- ; }
highstock-usage(){ cat << EOU

Highstock : time series plotting
==================================

http://www.highcharts.com/documentation/how-to-use

http://www.highcharts.com/download

http://chartit.shutupandship.com./


Hookup on WW
---------------

  highstock-
  highstock-get
  highstock-ln





EOU
}
highstock-dir(){ echo $(local-base)/env/plot/$(highstock-name) ; }
highstock-cd(){  cd $(highstock-dir); }
highstock-mate(){ mate $(highstock-dir) ; }
highstock-name(){ echo Highstock-1.1.6 ; } 
highstock-get(){
   local dir=$(dirname $(highstock-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(highstock-name)
   local zip=$nam.zip
   local url=http://code.highcharts.com/zips/$zip
   [ ! -f $zip ] && curl -L -O $url
   [ ! -d $nam ] && unzip $zip -d $nam

}

highstock-ln(){
   ln -s $(highstock-dir)/js $(env-home)/_static/highstock 
}



