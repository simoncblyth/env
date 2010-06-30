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

     Tried newer revisions of the slave ... but encounter 
     version incompatibility between slave and master 
    ... so back to the historical r561 in ghost branch 


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


[blyth@belle7 bitten]$ svn log slave.py
------------------------------------------------------------------------
r875 | osimons | 2010-05-14 18:51:22 +0800 (Fri, 14 May 2010) | 1 line

Merged [874] from trunk.
------------------------------------------------------------------------
r864 | wbell | 2010-04-30 02:19:48 +0800 (Fri, 30 Apr 2010) | 1 line

Port of [864] to 0.6.x
------------------------------------------------------------------------
r837 | wbell | 2010-04-24 22:30:06 +0800 (Sat, 24 Apr 2010) | 1 line

Port of [836] to 0.6.x
------------------------------------------------------------------------
r833 | wbell | 2010-04-24 21:38:25 +0800 (Sat, 24 Apr 2010) | 1 line

Merge of [832] from trunk.
------------------------------------------------------------------------
r800 | osimons | 2009-12-09 19:57:42 +0800 (Wed, 09 Dec 2009) | 1 line

Merge [797:799] from trunk.
------------------------------------------------------------------------




EOU
}
bitn-dir(){ echo $(local-base)/env/trac/bitn ; }
bitn-cd(){  cd $(bitn-dir); }
bitn-mate(){ mate $(bitn-dir) ; }


#bitn-branch(){ echo trunk ; }
#bitn-branch(){ echo branches/0.6.x ; }
#bitn-branch(){ echo tags/0.6b2 ; }
bitn-branch(){ echo branches/experimental/trac-0.11@561 ; }
bitn-url(){    echo http://svn.edgewall.org/repos/bitten/$(bitn-branch) ; }


bitn-get(){
   local dir=$(dirname $(bitn-dir)) &&  mkdir -p $dir && cd $dir
   local cmd="svn co $(bitn-url) bitn"
   echo $msg $cmd from $PWD
   eval $cmd
}


bitn-develop-slave(){
  local msg="=== $FUNCNAME :"
  bitn-cd

  ##  --without-master only present in newer revisions 
  local cmd="$SUDO python setup.py develop"
  echo $msg $cmd
  eval $cmd
}

