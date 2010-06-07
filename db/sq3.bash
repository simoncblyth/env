# === func-gen- : db/sq3 fgp db/sq3.bash fgn sq3 fgh db
sq3-src(){      echo db/sq3.bash ; }
sq3-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sq3-src)} ; }
sq3-vi(){       vi $(sq3-source) ; }
sq3-env(){      elocal- ; }
sq3-usage(){
  cat << EOU
     sq3-src : $(sq3-src)
     sq3-dir : $(sq3-dir)

   http://wanderinghorse.net/computing/sqlite/

EOU
}


sq3-name(){ echo libsqlite3x-2007.10.18 ; } 
sq3-url(){  echo http://wanderinghorse.net/computing/sqlite/$(sq3-name).tar.gz ; }

sq3-dir(){ echo $(local-base)/env/db/$(sq3-name) ; }
sq3-cd(){  cd $(sq3-dir); }
sq3-mate(){ mate $(sq3-dir) ; }
sq3-get(){
   local dir=$(dirname $(sq3-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -f "$(sq3-name).tar.gz" ] && curl -O $(sq3-url)
   [ ! -d "$(sq3-name)"  ]  && tar zxvf $(sq3-name).tar.gz 


}
