# === func-gen- : nuwa/scraper/scraper fgp nuwa/scraper/scraper.bash fgn scraper fgh nuwa/scraper
scraper-src(){      echo nuwa/scraper/scraper.bash ; }
scraper-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scraper-src)} ; }
scraper-vi(){       vi $(scraper-source) ; }
scraper-env(){      elocal- ; }
scraper-usage(){ cat << EOU





EOU
}
scraper-dir(){ echo $(local-base)/env/nuwa/Scraper ; }
scraper-cd(){  cd $(scraper-dir)/$1 ; }
scraper-mate(){ mate $(scraper-dir) ; }
scraper-get(){
   local dir=$(dirname $(scraper-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d Scraper ] && svn co http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/Database/Scraper 
}

scraper-dq(){ echo $(scraper-dir)/python/Scraper/dq ; }
scraper-dq-cd(){ cd $(scraper-dq) ; }




