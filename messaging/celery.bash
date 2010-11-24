# === func-gen- : messaging/celery fgp messaging/celery.bash fgn celery fgh messaging
celery-src(){      echo messaging/celery.bash ; }
celery-source(){   echo ${BASH_SOURCE:-$(env-home)/$(celery-src)} ; }
celery-vi(){       vi $(celery-source) ; }
celery-env(){      elocal- ; }
celery-usage(){
  cat << EOU
     celery-src : $(celery-src)
     celery-dir : $(celery-dir)

   For info on the command line tool for testing AMQP servers see 
       camqadm-;camqadm-vi

   Issue with camqadm when using yes/no options ( "too many values to unpack errors" )
   forces use of lastest celery from github
   
       https://github.com/ask/celery/issues/issue/257


EOU
}
celery-dir(){ echo $(local-base)/env/messaging/celery ; }
celery-cd(){  cd $(celery-dir); }
celery-mate(){ mate $(celery-dir) ; }
celery-get(){
   local dir=$(dirname $(celery-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/ask/celery.git
}

celery-build(){

   celery-cd
   python setup.py build
   python setup.py install

   pip install kombu    ## seems missing dependency 

}

