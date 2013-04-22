# === func-gen- : tools/svnlog fgp tools/svnlog.bash fgn svnlog fgh tools
svnlog-src(){      echo tools/svnlog.bash ; }
svnlog-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svnlog-src)} ; }
svnlog-vi(){       vi $(svnlog-source)  ; }
svnlog-env(){      elocal- ; }
svnlog-usage(){ cat << EOU

SVNLOG BASH FUNCTIONS
----------------------

Queries SVN for commit messages when invoked from SVN working copy

Usage::
          
   svnlog-
   svnlog -h

   svnlog -a blyth     

   svnlog-wcdirs-
          ## list of SVN working copy dirs
              
   svnlog-collect 
          ## collect last 52 weeks of commit messages from all svnlog-wcdirs- 


::

	[blyth@belle7 e]$ ll /tmp/env/svnlog-collect/
	total 312
	drwxrwxr-x 4 blyth blyth  4096 Apr 22 20:37 ..
	-rw-rw-r-- 1 blyth blyth 33388 Apr 22 20:37 env.txt
	-rw-rw-r-- 1 blyth blyth 61218 Apr 22 20:37 envd.txt
	-rw-rw-r-- 1 blyth blyth 18729 Apr 22 20:37 heprez.txt
	-rw-rw-r-- 1 blyth blyth 39400 Apr 22 20:37 heprezd.txt
	-rw-rw-r-- 1 blyth blyth 24644 Apr 22 20:43 dybgaudi.txt
	-rw-rw-r-- 1 blyth blyth 44540 Apr 22 20:44 dybgaudid.txt
	-rw-rw-r-- 1 blyth blyth 24644 Apr 22 20:45 dybinst.txt
	drwxrwxr-x 2 blyth blyth  4096 Apr 22 20:45 .
	-rw-rw-r-- 1 blyth blyth 44540 Apr 22 20:46 dybinstd.txt
	[blyth@belle7 e]$ 


.. warning:: a shim svnlog.py script has been added to ENV_HOME/bin/ largely removing the need for this svnlog- precursor 


EOU
}
svnlog-dir(){ echo $(local-base)/env/tools/tools-svnlog ; }
svnlog-cd(){  cd $(svnlog-dir); }
svnlog-mate(){ mate $(svnlog-dir) ; }
svnlog(){
   python $(env-home)/tools/svnlog.py $*
}


svnlog-wcdirs-(){  cat << EOD
$ENV_HOME
$HEPREZ_HOME
$DYB/NuWa-trunk/dybgaudi
$DYB/installation/trunk/dybinst
EOD
}

svnlog-collect(){
  local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
  local dir
  local name
  svnlog-wcdirs- | while read dir ; do
     [ ! -d "$dir" ] && echo $msg $dir does not exist && return 1
     cd $dir
     name=$(basename $dir) 
     echo
     echo $msg ============================== $name : $dir
     echo
     svn up $dir
     svnlog.py  --limit 1000000 -w 52 -a blyth > $tmp/${name}.txt 
     svnlog.py  --limit 1000000 -w 52 -a blyth --details  > $tmp/${name}d.txt 
  done
}

