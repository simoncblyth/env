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




auto-pulling from remote 
---------------------------

* https://docs.gitlab.com/ee/user/project/repository/repository_mirroring.html#pulling-from-a-remote-repository-starter


.gitlab-ci.yml
----------------

* https://docs.gitlab.com/ee/ci/quick_start/

Search github for some examples

* https://github.com/kassonlab/gmxapi/blob/master/.gitlab-ci.yml

* https://github.com/kassonlab/gmxapi/blob/master/admin/gitlab-ci/documentation.gitlab-ci.yml

  Yuck : surely better to just run some bash scripts, why translate into yml ? 


GitLab CI/CD pipeline configuration reference
-----------------------------------------------

* https://docs.gitlab.com/ee/ci/yaml/
* https://www.youtube.com/watch?v=Jav4vbUrqII&feature=emb_rel_end



EOU
}
gitlab-dir(){ echo $(local-base)/env/git/git-gitlab ; }
gitlab-cd(){  cd $(gitlab-dir); }
gitlab-mate(){ mate $(gitlab-dir) ; }
gitlab-get(){
   local dir=$(dirname $(gitlab-dir)) &&  mkdir -p $dir && cd $dir

}
