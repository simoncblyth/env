# === func-gen- : db/sqlite3x fgp db/sqlite3x.bash fgn sqlite3x fgh db
sqlite3x-src(){      echo db/sqlite3x.bash ; }
sqlite3x-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sqlite3x-src)} ; }
sqlite3x-vi(){       vi $(sqlite3x-source) ; }
sqlite3x-env(){      elocal- ; }
sqlite3x-usage(){
  cat << EOU
     sqlite3x-src : $(sqlite3x-src)
     sqlite3x-dir : $(sqlite3x-dir)


EOU
}

sqlite3x-name(){ echo sqlite3x-050616 ; }
sqlite3x-url(){  echo http://prdownloads.sourceforge.net/int64/$(sqlite3x-name).zip?download ; }

sqlite3x-dir(){ echo $(local-base)/env/db/sqlite3x ; }
sqlite3x-cd(){  cd $(sqlite3x-dir); }
sqlite3x-mate(){ mate $(sqlite3x-dir) ; }
sqlite3x-get(){
   local dir=$(dirname $(sqlite3x-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -f "$(sqlite3x-name).zip" ] && curl -o $(sqlite3x-name).zip -L $(sqlite3x-url)
   [ ! -d "$(sqlite3x-name)"     ] && unzip $(sqlite3x-name).zip 


}
