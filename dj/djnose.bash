# === func-gen- : dj/djnose fgp dj/djnose.bash fgn djnose fgh dj
djnose-src(){      echo dj/djnose.bash ; }
djnose-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djnose-src)} ; }
djnose-vi(){       vi $(djnose-source) ; }
djnose-env(){      elocal- ; }
djnose-usage(){
  cat << EOU
     djnose-src : $(djnose-src)
     djnose-dir : $(djnose-dir)


     http://wiki.github.com/jbalogh/django-nose/


g4pb:django-nose blyth$ git branch -r
  origin/HEAD -> origin/master
  origin/django-1.2
  origin/master

g4pb:django-nose blyth$ git branch -a
* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/django-1.2
  remotes/origin/master

g4pb:django-nose blyth$ git checkout -b django-1.2 remotes/origin/django-1.2
Branch django-1.2 set up to track remote branch django-1.2 from origin.
Switched to a new branch 'django-1.2'

g4pb:django-nose blyth$ git branch
* django-1.2
  master

http://stackoverflow.com/questions/67699/how-do-i-clone-all-remote-branches-with-git


  * needed to update django trunk in order for django-nose to find the test runner class  
    updated from 11453 to 12996


EOU
}
djnose-dir(){ echo $(local-base)/env/dj/django-nose ; }
djnose-cd(){  cd $(djnose-dir); }
djnose-mate(){ mate $(djnose-dir) ; }
djnose-url(){ echo git://github.com/jbalogh/django-nose.git ; }
djnose-get(){
   local dir=$(dirname $(djnose-dir)) &&  mkdir -p $dir && cd $dir
   git clone $(djnose-url) 
   cd django-nose
   git checkout -b django-1.2 remotes/origin/django-1.2

   sudo pip install -e .
}
