# === func-gen- : offline/scr fgp offline/scr.bash fgn scr fgh offline
scr-src(){      echo offline/scr.bash ; }
scr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scr-src)} ; }
scr-vi(){       vi $(scr-source) ; }
scr-env(){      
   elocal- 
   [ -d ~/v/scr ] && . ~/v/scr/bin/activate
}
scr-usage(){
  cat << EOU
     scr-src : $(scr-src)
     scr-dir : $(scr-dir)

    GENERIC SCRAPER INVESTIGATIONS 

   N 
      SQLAlchemy-0.6.7.tar.gz
      ipython-0.10.2      

EOU
}
scr-dir(){ echo $(local-base)/env/offline/offline-scr ; }
scr-cd(){  cd $(scr-dir); }
scr-mate(){ mate $(scr-dir) ; }
scr-get(){
   local dir=$(dirname $(scr-dir)) &&  mkdir -p $dir && cd $dir

   if [ ! -d "~/v/scr" ]; then  
       mkdir -p ~/v   
       virtualenv ~/v/scr
       . ~/v/scr/bin/activate
       pip install sqlalchemy 
       pip install --upgrade ipython
   fi

}

scr-check(){
   python -c "import sqlalchemy as sa  ; print sa.__version__ "
}
