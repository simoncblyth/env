# === func-gen- : sa/rum fgp sa/rum.bash fgn rum fgh sa
rum-src(){      echo rum/rum.bash ; }
rum-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rum-src)} ; }
rum-vi(){       vi $(rum-source) ; }


rum-env-C(){
   ## system python on cms01 : 2.3.4 is too old to use ... so the basis of the virtualenv is the source python
   python- source 
}

rum-mode(){ echo ${RUM_MODE:-dev} ; }
rum-env(){      
   elocal- ; 
   case $NODE_TAG in  
     C) rum-env-C ;;
   esac
   rum-activate ;  

   ## this distinguishes deployed running and debug running 
   case $(rum-mode) in 
     dev) export ENV_PRIVATE_PATH=$HOME/.bash_private ;; 
       *) export ENV_PRIVATE_PATH=$(apache-private-path) ;;
   esac
}
rum-usage(){
  cat << EOU
     rum-src : $(rum-src)
     rum-dir : $(rum-dir)

    Pre-requisites :
        mysql server running, 
                mysql-start

    rum-/rum-env
        activates the rumenv virtual python and 
        distinguishes dev and deployment modes via the 
        setting of ENV_PRIVATE_PATH 
           
           rum-
           echo $ENV_PRIVATE_PATH
              /home/blyth/.bash_private

           RUM_MODE=deploy rum-
           echo $ENV_PRIVATE_PATH
              /data1/env/local/env/.bash_private

    rum-get
        creates and gets into the virtual python
        (rum installation now down with vdbi-)

    rum-tute 
        run the tutorial
        view the app at :  http://localhost:8080

       for interactive investigation from ipython use :
           %run -d app.py
       

    For bumping some the components (eg tw.rum) up to their tips ... see 
        rumdev-install 


    http://docs.python-rum.org/user/install.html

EOU
}
rum-dir(){ echo $(local-base)/env/rumenv ; }
rum-cd(){  cd $(rum-dir); }
rum-mate(){ mate $(rum-dir) ; }


rum-build(){
   ! rum-get && return 1
   env-build
}

rum-wipe(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo rm -rf $(rum-dir) " 
   local ans
   read -p "$msg proceed with \"$cmd\"  enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipped && return 0
   eval $cmd
}

rum-get(){
   local dir=$(dirname $(rum-dir)) &&  mkdir -p $dir && cd $dir
   local msg="=== $FUNCNAME :"
   [ "$(which virtualenv)" == "" ] && echo $msg missing virtualenv && return 1
   [ ! -d "$(rum-dir)" ] && virtualenv $(rum-dir) || echo $msg virtualenv dir $(rum-dir) exists already skipping virtualenv creation 
   ! rum-activate && echo "$msg failed to activate " && return 1 
   [ "$(which python)" != "$(rum-dir)/bin/python" ] && echo $msg ABORT failed to setup virtual python $(which python)   && return 1
   [ "$(python -c 'import MySQLdb')" != ""  ] && echo $msg ABORT missing MySQLdb && return 1   
   return 0
}



rum-activate(){   
  local activate=$(rum-dir)/bin/activate 
  . $activate 
}
rum-deactivate(){ deactivate ; }

rum-tute(){
   cd $(env-home)/rum/tutorial
   python app.py
}


