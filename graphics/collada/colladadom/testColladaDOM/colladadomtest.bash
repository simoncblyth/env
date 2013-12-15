colladadomtest-src(){      echo graphics/collada/colladadom/testColladaDOM/colladadomtest.bash ; }
colladadomtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(colladadomtest-src)} ; }
colladadomtest-vi(){       vi $(colladadomtest-source) ; }
colladadomtest-env(){      elocal- ; }
colladadomtest-usage(){ cat << EOU

ColladaDOM test
=================

* https://collada.org/mediawiki/index.php/DOM_guide:_Working_with_documents



EOU
}
colladadomtest-dir(){ echo $(env-home)/graphics/collada/colladadom/testColladaDOM ; }
colladadomtest-cd(){  cd $(colladadomtest-dir); }
colladadomtest-mate(){ mate $(colladadomtest-dir) ; }
colladadomtest-get(){
   local dir=$(dirname $(colladadomtest-dir)) &&  mkdir -p $dir && cd $dir

}
