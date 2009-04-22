so-src(){ echo so/so.bash ; }
so-source(){ echo ${BASH_SOURCE:-$(env-home)/$(so-src)} ; }
so-vi(){     vi $(so-source); }
so-env(){    echo -n ; }
so-usage(){
  cat << EOU

    so-dangle
         list broken links in $(so-dir) 

    so-date
          
EOU

}


so-base(){ echo ${DDR:-.} ; }
so-dir(){  echo  $(so-base)/dybgaudi/InstallArea/$CMTCONFIG/lib/ ; }

so-abs(){  echo $(dirname $1)/$(readlink $1) ; }
so-date(){
	local so
	for so in $(so-dir)/*.so ; do
	    if [ -L $so ]; then
	     	local aso=$(so-abs $so)
	     	[ -f $aso ] && ls -l $aso
	    fi
	done	
}
so-dangle(){
	local so
	for so in $(so-dir)/*.so ; do
		if [ -L $so ]; then
			local aso=$(so-abs $so)
			[ ! -f $aso ] && echo $so 
		fi
	done
}


