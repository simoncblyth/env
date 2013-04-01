# === func-gen- : git/gitlab fgp git/gitlab.bash fgn gitlab fgh git
gitlab-src(){      echo git/gitlab.bash ; }
gitlab-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitlab-src)} ; }
gitlab-vi(){       vi $(gitlab-source) ; }
gitlab-env(){      elocal- ; }
gitlab-usage(){ cat << EOU

GitLab : self hosted git, like github
========================================

https://dev.gitlab.org/public

https://github.com/gitlabhq/gitlabhq/blob/master/README.md



EOU
}
gitlab-dir(){ echo $(local-base)/env/git/git-gitlab ; }
gitlab-cd(){  cd $(gitlab-dir); }
gitlab-mate(){ mate $(gitlab-dir) ; }
gitlab-get(){
   local dir=$(dirname $(gitlab-dir)) &&  mkdir -p $dir && cd $dir

}
