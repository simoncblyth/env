# === func-gen- : base/time/realtime fgp base/time/realtime.bash fgn realtime fgh base/time
realtime-src(){      echo base/time/realtime.bash ; }
realtime-source(){   echo ${BASH_SOURCE:-$(env-home)/$(realtime-src)} ; }
realtime-vi(){       vi $(realtime-source) ; }
realtime-env(){      elocal- ; }
realtime-usage(){ cat << EOU

Realtime
=========

* http://nadeausoftware.com/articles/2012/04/c_c_tip_how_measure_elapsed_real_time_benchmarking




EOU
}
realtime-dir(){ echo $(env-home)/base/time ; }
realtime-cd(){  cd $(realtime-dir); }
realtime-mate(){ mate $(realtime-dir) ; }
realtime-get(){
   local dir=$(realtime-dir) &&  mkdir -p $dir && cd $dir
   local url="http://nadeausoftware.com/sites/NadeauSoftware.com/files/getRealTime.c"
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -L -O $url

}
