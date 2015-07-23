# === func-gen- : tools/java/java fgp tools/java/java.bash fgn java fgh tools/java
java-src(){      echo tools/java/java.bash ; }
java-source(){   echo ${BASH_SOURCE:-$(env-home)/$(java-src)} ; }
java-vi(){       vi $(java-source) ; }
java-env(){      elocal- ; }
java-usage(){ cat << EOU


JAVA
======


OSX Mavericks Install Oracle Java (Delta)
-------------------------------------------

* http://www.java.com/en/download/

Version 8 Update 51
Release date July 14, 2015 

Web download yields::
   
    /Users/blyth/Downloads/jre-8u51-macosx-x64.dmg

Opening DMG gives::

    /Volumes/Java\ 8\ Update\ 51/Java\ 8\ Update\ 51.pkg 

Record contents of pkg before installing::

    java-pkgpath-lsbom > ~/packages/java-pkgpath-lsbom.txt

After install and allowing Java plugin to run for java.com find::

   https://java.com/en/download/installed.jsp




EOU
}
java-dir(){ echo $(local-base)/env/tools/java/tools/java-java ; }
java-cd(){  cd $(java-dir); }
java-mate(){ mate $(java-dir) ; }
java-get(){
   local dir=$(dirname $(java-dir)) &&  mkdir -p $dir && cd $dir

}


java-pkgpath(){
   echo "/Volumes/Java 8 Update 51/Java 8 Update 51.pkg"
}

java-pkgpath-lsbom()
{
   lsbom "$(pkgutil --bom "$(java-pkgpath)")"
}


