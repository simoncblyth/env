# === func-gen- : git/getgitorious fgp git/getgitorious.bash fgn getgitorious fgh git
getgitorious-src(){      echo git/getgitorious.bash ; }
getgitorious-source(){   echo ${BASH_SOURCE:-$(env-home)/$(getgitorious-src)} ; }
getgitorious-vi(){       vi $(getgitorious-source) ; }
getgitorious-env(){      elocal- ; }
getgitorious-usage(){ cat << EOU

Gitorious : Hosting Git Repos
=================================

* http://getgitorious.com/installer#sec-2
* http://www.barryodonovan.com/index.php/2013/01/04/git-web-applications-aka-github-alternatives

Intstaller is for CentOS, would need mods for Redhat,
depends on ruby and ruby on rails
On N (C probably too old to even test), would need::

      sudo yum --enablerepo=epel search ruby

EOU
}
getgitorious-dir(){ echo $(local-base)/env/git/getgitorious ; }
getgitorious-cd(){  cd $(getgitorious-dir); }
getgitorious-mate(){ mate $(getgitorious-dir) ; }
getgitorious-get(){
   local dir=$(dirname $(getgitorious-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://gitorious.org/gitorious/ce-installer.git && cd ce-installer


}
