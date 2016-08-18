# === func-gen- : juno/juno fgp juno/juno.bash fgn juno fgh juno
juno-src(){      echo juno/juno.bash ; }
juno-source(){   echo ${BASH_SOURCE:-$(env-home)/$(juno-src)} ; }
juno-vi(){       vi $(juno-source) ; }
juno-env(){      elocal- ; }
juno-usage(){ cat << EOU


JUNO
=====


Saving SSO Passwd in browser cache
---------------------------------------

Some of us feel a bit inconvenient that we need to input the JUNO website
credentials each time we come back to the site. Speculation is that autosave
functionality is disabled for the users. Is it on purpose? Possible to switch
it back? 

Thanks. Wei

Here is a solution. Please use the SSO login URL directly and then you can save your password:
https://idp.ihep.ac.cn/idp/Authn/UserPassword 
There will be an error on the web after your login. Please ignore the error.

BTW, this is only used for password saving purpose.
Except that please DO NOT access the SSO login URL directly, 
for there will be an error page returned all the time.

For those who want to know the reason, a simple explanation is that the SSO authentication is a bit complex.
Take the DocDB as example, there will be 2 redirections during your login, 
DocDB -> SSO IdP Server -> DocDB. 
The password saving dialog is arised at SSO IdP Server, 
but it is flushed out by the 2nd redirection automatically.
So my solution is accessing the SSO IdP Server (SSO login RUL) directly. 
Then the system will not know where to redirect, 
and the password saving dialog will be kept. 
As a result, there will be an error notification.

Best regards,
Jiaheng Zou
Computing Center, Institute of High Energy Physics
Chinese Academy of Sciences




EOU
}
juno-dir(){ echo $(local-base)/env/juno/juno-juno ; }
juno-cd(){  cd $(juno-dir); }
juno-mate(){ mate $(juno-dir) ; }
juno-get(){
   local dir=$(dirname $(juno-dir)) &&  mkdir -p $dir && cd $dir

}
