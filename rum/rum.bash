# === func-gen- : sa/rum fgp sa/rum.bash fgn rum fgh sa
rum-src(){      echo rum/rum.bash ; }
rum-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rum-src)} ; }
rum-vi(){       vi $(rum-source) ; }
rum-env(){      elocal- ; rum-activate ;  }
rum-usage(){
  cat << EOU
     rum-src : $(rum-src)
     rum-dir : $(rum-dir)


     rum-tute 
        run the tutorial
        view the app at :  http://localhost:8080

       for interactive investigation from ipython use :
           %run -d app.py
       
    rum-get
        pre-requisite : virtualenv, get that with virtualenv-get

     For bumping some the components (eg tw.rum) up to their tips ... see 
        rumdev-install 


    http://docs.python-rum.org/user/install.html

EOU
}
rum-dir(){ echo $(local-base)/env/rumenv ; }
rum-cd(){  cd $(rum-dir); }
rum-mate(){ mate $(rum-dir) ; }
rum-get(){
   local dir=$(dirname $(rum-dir)) &&  mkdir -p $dir && cd $dir
   [ "$(which virtualenv)" == "" ] && echo $msg missing virtualenv && return 1
   [ ! -d "$(rum-dir)" ] && virtualenv $(rum-dir) || echo $msg virtualenv dir $(rum-dir) exists already skipping virtualenv creation 
   rum-activate
   [ "$(which easy_install)" == "/usr/bin/easy_install" ] && echo $msg failed to setup virtualenv && return 1

   
   ## avoid getting stymied by directory name clashes, by moving to an empty dir ... avoids : 
   ##    error: Couldn't find a setup script in /usr/local/env/ipython

   local iwd=$PWD
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp && cd $tmp
   
   easy_install rum RumAlchemy tw.rum 
   easy_install ipython
   cd $iwd
}
rum-activate(){   . $(rum-dir)/bin/activate ; }
rum-deactivate(){ deactivate ; }
rum-projdir(){ echo $(dirname $(rum-source))/rum ; }
rum-tute(){
   cd $(rum-projdir)/tutorial
   python app.py
}



