# === func-gen- : plot/svgcharts/cqpack fgp plot/svgcharts/cqpack.bash fgn cqpack fgh plot/svgcharts
cqpack-src(){      echo plot/svgcharts/cqpack.bash ; }
cqpack-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cqpack-src)} ; }
cqpack-vi(){       vi $(cqpack-source) ; }
cqpack-env(){      elocal- ; }
cqpack-usage(){ cat << EOU

CQPACK : plotting CQ packing times
====================================


FUNCTIONS
----------

*cqpack-makesvg*
       grab numbers by grepping some remote logfiles, massage into python dict strings 
       with a perl oneliner, convert into SVG plot

*cqpack-open*
       plant symbolic link to SVG output directory in apache htdocs and open browser 
       on that url 


EOU
}
cqpack-dir(){ echo $(local-base)/env/plot/svgcharts/cqpack ; }
cqpack-cd(){  cd $(cqpack-dir); }
cqpack-mate(){ mate $(cqpack-dir) ; }


cqpack-makesvg(){
  local dir=$(cqpack-dir) ; mkdir -p $dir

  ssh N grep -H seconds: /data1/env/local/dyb/NuWa-trunk/logs/CQCatchup/\*.log > $dir/seconds.txt
  perl -n -e 'm,_(\d{4}).log.*{(.*)}, && print "{ \"index\":\"$1\",  $2 }\n" ' $dir/seconds.txt > $dir/seconds.dat

  $(env-home)/plot/svgcharts/cqpack.py $dir/seconds.dat 
}

cqpack-open(){
  local dir=$(cqpack-dir) ; mkdir -p $dir
  apache-
  ( cd `apache-htdocs` && rm cqpack && ln -s $dir cqpack )
  open http://localhost/cqpack/
}
