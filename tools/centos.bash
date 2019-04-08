# === func-gen- : tools/centos fgp tools/centos.bash fgn centos fgh tools src base/func.bash
centos-source(){   echo ${BASH_SOURCE} ; }
centos-edir(){ echo $(dirname $(centos-source)) ; }
centos-ecd(){  cd $(centos-edir); }
centos-dir(){  echo $LOCAL_BASE/env/tools/centos ; }
centos-cd(){   cd $(centos-dir); }
centos-vi(){   vi $(centos-source) ; }
centos-env(){  elocal- ; }
centos-usage(){ cat << EOU


* https://danielmiessler.com/study/fedora_redhat_centos/


Fedora is the main project, and it’s a communitity-based, free distro
focused on quick releases of new features and functionality.

Redhat is the corporate version based on the progress of that project, and
it has slower releases, comes with support, and isn’t free.

CentOS is basically the community version of Redhat. So it’s pretty much
identical, but it is free and support comes from the community as opposed to
Redhat itself.

* https://en.wikipedia.org/wiki/CentOS

Relation between RHEL version and CentOS is one-to-one



=========  ======= =====  ============  ==================  =====================  ============
CentOS      arch   RHEL    Kernel        Centos release       RHEL release          days delay
=========  ======= =====  ============  ==================  =====================  ============
7.0-1406 	x86-64  7.0 	3.10.0-123 	7 July 2014 	     10 June 2014 	        27
7.1-1503 	x86-64 	7.1 	3.10.0-229 	31 March 2015  	     5 March 2015 	        26
7.2-1511 	x86-64 	7.2 	3.10.0-327 	14 December 2015 	 19 November 2015    	25
7.3-1611 	x86-64 	7.3 	3.10.0-514 	12 December 2016 	 3 November 2016    	39
7.4-1708 	x86-64 	7.4 	3.10.0-693 	13 September 2017 	 31 July 2017           43
7.5-1804 	x86-64 	7.5 	3.10.0-862 	10 May 2018     	 10 April 2018          31
7.6-1810 	x86-64 	7.6 	3.10.0-957 	3 December 2018 	 30 October 2018        34 
=========  ======= =====  ============  ==================  =====================  ============




EOU
}
centos-get(){
   local dir=$(dirname $(centos-dir)) &&  mkdir -p $dir && cd $dir

}
