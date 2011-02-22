# === func-gen- : messaging/pika fgp messaging/pika.bash fgn pika fgh messaging
pika-src(){      echo messaging/pika.bash ; }
pika-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pika-src)} ; }
pika-vi(){       vi $(pika-source) ; }
pika-env(){      elocal- ; }
pika-usage(){
  cat << EOU
     pika-src : $(pika-src)
     pika-dir : $(pika-dir)
     
     http://github.com/tonyg/pika

     To test operation ... 
         1) in one session : pika-consume
         2) in another     : pika-send

      The sessions can be on different machines but the 
      AMQP_ config for the sessions has to point 
      to the same vhost in the same server (or erlang cluster i suppose)

     Usage example ...

          pika-consume --help
          pika-consume -x testolive
             fails noisily if exchange does not exist 


     Some help with git ..

         http://help.github.com/git-cheat-sheets/
         http://www.gitready.com/beginner/2009/03/09/remote-tracking-branches.html


          pika-gitsetup 
               create 0_9_1 remote tracking branch

          pika-update
                update from remote via "git pull"

          pika-091
          pika-080
                switch to pika branch supporting AMQP 0_9_1 or 0_8

         pika-info 
                dump branch info 


    On G, Feb 2011 with the ~/v/mq virtualenv

         pika-update
         python setup.py build
         python setup.py install     ## no sudo as using virtualenv


EOU
}
pika-dir(){ echo $(local-base)/env/messaging/pika ; }
pika-cd(){  cd $(pika-dir); }
pika-scd(){  cd $(env-home)/messaging/pika ; }
pika-mate(){ mate $(pika-dir) ; }

pika-wipe(){
   local dir=$(dirname $(pika-dir)) &&  mkdir -p $dir && cd $dir
   rm -rf pika 
}

pika-get(){
   local dir=$(dirname $(pika-dir)) &&  mkdir -p $dir && cd $dir
   git clone http://github.com/tonyg/pika.git
}

pika-ln(){
  python-
  python-ln $(pika-dir)/pika
}

pika-gitsetup(){
   pika-cd
   git branch --track amqp_0_9_1  origin/amqp_0_9_1
   git branch -a 
}

pika-091(){
   pika-cd
   git checkout amqp_0_9_1
   pika-info
}

pika-080(){
   pika-cd
   git checkout master 
   pika-info
}

pika-update(){
   pika-cd
   git pull
}

pika-info(){
   local msg="=== $FUNCNAME :"
   pika-cd
   echo $msg local branches ..
   git branch 
   echo $msg remote branches
   git branch -r
   echo $msg all branches
   git branch -a

}


pika-send(){    python $(env-home)/messaging/pika/send.py $* ; }
pika-consume(){ python $(env-home)/messaging/pika/consume.py $* ; }

pika-v(){ echo ~/v/mq ; }
pika-activate(){
   . $(pika-v)/bin/activate
}
pika-mon(){
   $(pika-v)/bin/python $(env-home)/messaging/pika/mon.py $* 
}
pika-imon(){
   . $(pika-v)/bin/activate
   cd $(env-home)/messaging/pika
   ipython
}
pika-i(){
   $(pika-v)/bin/ipython 
}


