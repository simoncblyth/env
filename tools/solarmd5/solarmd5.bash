# === func-gen- : tools/solarmd5/solarmd5 fgp tools/solarmd5/solarmd5.bash fgn solarmd5 fgh tools/solarmd5
solarmd5-src(){      echo tools/solarmd5/solarmd5.bash ; }
solarmd5-source(){   echo ${BASH_SOURCE:-$(env-home)/$(solarmd5-src)} ; }
solarmd5-vi(){       vi $(solarmd5-source) ; }
solarmd5-env(){      elocal- ; }
solarmd5-usage(){ cat << EOU

Public Domain Drop in for MD5 from Solar Designer
===================================================

http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5


EOU
}
solarmd5-dir(){ echo $(local-base)/env/tools/solarmd5 ; }
solarmd5-cd(){  cd $(solarmd5-dir); }
solarmd5-urlbase(){ echo http://openwall.info/wiki/_media/people/solar/software/public-domain-source-code ; }
solarmd5-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(solarmd5-dir) &&  mkdir -p $dir && cd $dir
   local urlbase=$(solarmd5-urlbase)
   local names="md5.h md5.c"
   local name
   for name in $names ; do
       if [ ! -f $name ]; then
          echo $msg downloading $urlbase/$name
          curl -L -O $urlbase/$name
       else
          echo $msg already downloaded $urlbase/$name
       fi  
   done
}

solarmd5-copyhere(){ cat << EOC
echo change directory then pipe these commands to sh if you want
cp $(solarmd5-dir)/md5.h $PWD/md5.h
cp $(solarmd5-dir)/md5.c $PWD/md5.c
EOC
}

