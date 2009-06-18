# === func-gen- : aberdeen/abd.bash fgp aberdeen/abd.bash fgn abd
abd-src(){      echo aberdeen/abd.bash ; }
abd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(abd-src)} ; }
abd-vi(){       vi $(abd-source) ; }
abd-env(){      elocal- ; }
abd-usage(){
  cat << EOU
     abd-src : $(abd-src)

  http://theta13.phy.cuhk.edu.hk/elog/Aberdeen-calibration/57

  cat Run1/README
  116820001 2009-06-16 20:17 run00015.mid  No Source
  626869026 2009-06-16 20:18 run00016.mid  Pedestal
  155052959 2009-06-16 20:18 run00017.mid  Ring 1
  156723488 2009-06-16 20:18 run00018.mid  Ring 2
  156638044 2009-06-16 20:19 run00019.mid  Center
  156490906 2009-06-16 20:19 run00020.mid  Ring 3
  150392026 2009-06-16 20:19 run00021.mid  Ring 4

  cat Run2/README
  100612139 2009-06-17 16:48 run00023.mid  No Source
   72835623 2009-06-17 16:48 run00024.mid  Pedestal
  136470954 2009-06-17 16:48 run00025.mid  Ring 1
  139572582 2009-06-17 16:48 run00026.mid  Ring 2
  139055374 2009-06-17 16:48 run00027.mid  Center
  139228840 2009-06-17 16:48 run00028.mid  Ring 3
  134263919 2009-06-17 16:48 run00029.mid  Ring 4

EOU
}

abd-names(){ echo Run1.tar.gz Run2.tar.gz ; }
abd-url(){   echo http://theta13.phy.cuhk.edu.hk/~aberdeen ; }
abd-dir(){   echo $(local-base)/env/abd ; }

abd-get(){
   local dir=$(abd-dir)
   mkdir -p $dir && cd $dir
   local name ; for name in $(abd-names) ;  do
      [ ! -f "$name" ] && curl -O $(abd-url)/$name
      local base=${name/.*}
      [ ! -d "$base" ] && mkdir -p "$base" && tar -C "$base" -zxvf $name 
   done
}


