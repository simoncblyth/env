# === func-gen- : http/jmeter fgp http/jmeter.bash fgn jmeter fgh http
jmeter-src(){      echo http/jmeter.bash ; }
jmeter-source(){   echo ${BASH_SOURCE:-$(env-home)/$(jmeter-src)} ; }
jmeter-vi(){       vi $(jmeter-source) ; }
jmeter-env(){      elocal- ; }
jmeter-usage(){
  cat << EOU
     jmeter-src : $(jmeter-src)
     jmeter-dir : $(jmeter-dir)

     http://jakarta.apache.org/jmeter/usermanual/index.html
EOU
}

jmeter-name(){ echo jakarta-jmeter-2.3.4 ; }
jmeter-url(){  echo http://ftp.twaren.net/Unix/Web/apache/jakarta/jmeter/binaries/$(jmeter-name).zip ; }
jmeter-docs(){ open $(jmeter-dir)/printable_docs/index.html ; }
jmeter-dir(){ echo $(local-base)/env/http/$(jmeter-name) ; }
jmeter-cd(){  cd $(jmeter-dir); }
jmeter-mate(){ mate $(jmeter-dir) ; }
jmeter-bin(){ echo $(jmeter-dir)/bin/jmeter ; }
jmeter-get(){
   local dir=$(dirname $(jmeter-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -f "$(jmeter-name).zip" ] && curl -L -O $(jmeter-url)
   [ ! -d "$(jmeter-name)"     ] && unzip $(jmeter-name).zip

   chmod u+x $(jmeter-bin)
   

}
jmeter-run(){ $(jmeter-bin) $* ; }


