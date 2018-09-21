# === func-gen- : software/carpentry fgp software/carpentry.bash fgn carpentry fgh software src base/func.bash
carpentry-source(){   echo ${BASH_SOURCE} ; }
carpentry-edir(){ echo $(dirname $(carpentry-source)) ; }
carpentry-ecd(){  cd $(carpentry-edir); }
carpentry-dir(){  echo $LOCAL_BASE/env/software/carpentry ; }
carpentry-cd(){   cd $(carpentry-dir); }
carpentry-vi(){   vi $(carpentry-source) ; }
carpentry-env(){  elocal- ; }
carpentry-usage(){ cat << EOU

* https://software-carpentry.org/about/

* https://f1000research.com/articles/3-62/v2

2.1 Version 1: red light

In 1995–96, the author organized a series of articles in IEEE Computational
Science & Engineering titled, “What Should Computer Scientists Teach to
Physical Scientists and Engineers?”9. These grew out of the frustration he had
working with scientists who wanted to run before they could walk, i.e., to
parallelize complex programs that were not broken down into self-contained
functions, that did not have any automated tests, and that were not under
version control10.

  Best Practices for Scientific Computing   

   * https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745





EOU
}
carpentry-get(){
   local dir=$(dirname $(carpentry-dir)) &&  mkdir -p $dir && cd $dir

}
