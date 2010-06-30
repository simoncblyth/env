# === func-gen- : trac/bitn fgp trac/bitn.bash fgn bitn fgh trac
bitn-src(){      echo trac/bitn.bash ; }
bitn-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bitn-src)} ; }
bitn-vi(){       vi $(bitn-source) ; }
bitn-env(){      elocal- ; }
bitn-usage(){
  cat << EOU
     bitn-src : $(bitn-src)
     bitn-dir : $(bitn-dir)

   TAKE A LOOK AT THE LATEST BITTEN-SLAVE 


   == belle7 : historically have been using 0.6dev_r561 with increased timout 15s patch == 

   == belle7 : try trunk slave : 0.7dev-r880 : FAILS incompatibility with master ?  ==

       removed the prior Bitten ...
          rm -rf Bitten-0.6dev_r561-py2.4.egg
          sudo vi easy-install.pth 
          [blyth@belle7 site-packages]$ bitten-slave --version
          bitten-slave 0.7dev-r880
{{{
[INFO    ] Build step export completed successfully
[DEBUG   ] Sending POST request to 'http://dayabay.ihep.ac.cn/tracs/dybsvn/builds/3476/steps/'
[DEBUG   ] Server returned error 500: Internal Server Error (no message available)
[ERROR   ] Exception raised processing step export. Reraising HTTP Error 500: Internal Server Error
[DEBUG   ] Stopping keepalive thread
[DEBUG   ] Keepalive thread exiting.
[DEBUG   ] Keepalive thread stopped
[ERROR   ] HTTP Error 500: Internal Server Error
[INFO    ] Slave exited at 2010-06-29 16:05:43
}}}

   Keepalive was introduced
      * http://bitten.edgewall.org/changeset/863

   A slave fix to do with authentication fails 
      * http://bitten.edgewall.org/ticket/330
      * http://bitten.edgewall.org/changeset/643


   == belle7 : try branches/0.6.x slave ==



EOU
}
bitn-dir(){ echo $(local-base)/env/trac/bitn ; }
bitn-cd(){  cd $(bitn-dir); }
bitn-mate(){ mate $(bitn-dir) ; }


bitn-rev(){ echo HEAD ; } 
#bitn-url(){ echo http://svn.edgewall.org/repos/bitten/trunk/ ; }
bitn-url(){ echo http://svn.edgewall.org/repos/bitten/branches/0.6.x/ ; }
bitn-get(){
   local dir=$(dirname $(bitn-dir)) &&  mkdir -p $dir && cd $dir
   svn co $(bitn-url)@$(bitn-rev) bitn
}


bitn-develop-slave(){
  local msg="=== $FUNCNAME :"
  bitn-cd
  local cmd="$SUDO python setup.py --without-master develop"
  echo $msg $cmd
  eval $cmd
}

