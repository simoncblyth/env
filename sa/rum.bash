# === func-gen- : sa/rum fgp sa/rum.bash fgn rum fgh sa
rum-src(){      echo sa/rum.bash ; }
rum-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rum-src)} ; }
rum-vi(){       vi $(rum-source) ; }
rum-env(){      elocal- ; }
rum-usage(){
  cat << EOU
     rum-src : $(rum-src)
     rum-dir : $(rum-dir)


     rum-tute 
        run the tutorial
        view the app at :  http://localhost:8080

       for interactive investigation from ipython use :
           %run -d app.py
       


    http://docs.python-rum.org/user/install.html

EOU
}
rum-dir(){ echo $(local-base)/env/sa/rum ; }
rum-cd(){  cd $(rum-dir); }
rum-mate(){ mate $(rum-dir) ; }
rum-get(){
   local dir=$(dirname $(rum-dir)) &&  mkdir -p $dir && cd $dir
   [ "$(which virtualenv)" == "" ] && echo $msg missing virtualenv && return 1
   [ ! -d "$(rum-dir)" ] && virtualenv $(rum-dir)
   rum-activate
   which easy_install

   easy_install rum RumAlchemy tw.rum ipython
}
rum-activate(){
   rum-cd
   . bin/activate
}
rum-deactivate(){
   rum-cd
   . bin/deactivate
}

rum-projdir(){ echo $(dirname $(rum-source))/rum ; }

rum-tute(){
   cd $(rum-projdir)/tutorial
   python app.py
   

}
