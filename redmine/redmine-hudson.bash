# === func-gen- : redmine/redmine-hudson fgp redmine/redmine-hudson.bash fgn redmine-hudson fgh redmine
redmine-hudson-src(){      echo redmine/redmine-hudson.bash ; }
redmine-hudson-source(){   echo ${BASH_SOURCE:-$(env-home)/$(redmine-hudson-src)} ; }
redmine-hudson-vi(){       vi $(redmine-hudson-source) ; }
redmine-hudson-env(){      elocal- ; }
redmine-hudson-usage(){
  cat << EOU
     redmine-hudson-src : $(redmine-hudson-src)
     redmine-hudson-dir : $(redmine-hudson-dir)


 == Redmine-Hudson ==


    http://www.r-labs.org/wiki/r-labs/Hudson_En/ 
    http://hudson.r-labs.org/hudson/
    http://www.r-labs.org/projects/hudson/repository


    https://github.com/agallou/hudson-redminecodenavigator-plugin


  Redmine-Hudson  (unofficial forks ?)
     https://github.com/dcramer/redmine_hudson
     https://github.com/AlekSi/redmine_hudson
     https://github.com/alsemyonov/redmine_hudson

  Official repo at rev 562
     http://r-labs.googlecode.com/svn/trunk/plugins/redmine_hudson/
  

  Very long comment threads ...
     http://www.redmine.org/boards/3/topics/6650
     http://www.redmine.org/boards/3/topics/14348  redmine-hudson 1.0.5


EOU
}
redmine-hudson-name(){ echo redmine-hudson ; } 
redmine-hudson-dir(){ echo $(local-base)/env/redmine/$(redmine-hudson-name) ; }
redmine-hudson-cd(){  cd $(redmine-hudson-dir); }
redmine-hudson-mate(){ mate $(redmine-hudson-dir) ; }
redmine-hudson-get(){
   local dir=$(dirname $(redmine-hudson-dir)) &&  mkdir -p $dir && cd $dir
   svn co http://r-labs.googlecode.com/svn/trunk/plugins/redmine_hudson/ $(redmine-hudson-name)
}
