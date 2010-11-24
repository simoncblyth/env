# === func-gen- : messaging/camqadm fgp messaging/camqadm.bash fgn camqadm fgh messaging
camqadm-src(){      echo messaging/camqadm.bash ; }
camqadm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(camqadm-src)} ; }
camqadm-vi(){       vi $(camqadm-source) ; }
camqadm-env(){      elocal- ; }
camqadm-usage(){
  cat << EOU
     camqadm-src : $(camqadm-src)
     camqadm-dir : $(camqadm-dir)

     camqadm is an AMQP command line client that comes with recent celery ...
     to get it upgrade celery with 

        pip install --upgrade celery   

    http://ask.github.com/celery/reference/celery.bin.camqadm.html


        camqadm--
             invoke camqadm CLI with messaging/camqadm on PYTHONPATH
             in order to find the celeryconfig.py file 
             which uses the private config 
            


EOU
}
camqadm-dir(){ echo $(local-base)/env/messaging/messaging-camqadm ; }
camqadm-cd(){  cd $(camqadm-dir); }
camqadm-mate(){ mate $(camqadm-dir) ; }
camqadm-get(){
   local dir=$(dirname $(camqadm-dir)) &&  mkdir -p $dir && cd $dir
}


camqadm--(){
   PYTHONPATH=$(env-home)/messaging/camqadm camqadm $* 
}
