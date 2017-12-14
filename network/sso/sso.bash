# === func-gen- : network/sso/sso fgp network/sso/sso.bash fgn sso fgh network/sso
sso-src(){      echo network/sso/sso.bash ; }

sso-sdir(){ echo $(env-home)/network/sso ; }
sso-scd(){ cd $(sso-sdir) ; }

sso-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sso-src)} ; }
sso-vi(){       vi $(sso-source) ; }
sso-env(){      elocal- ; }
sso-usage(){ cat << EOU

SSO Debugging
================

CERN SSO and IHEP SSO both giving problems from Taiwan ??

* http://myitforum.com/myitforumwp/2014/01/13/the-so-annoying-msis7000-signin-request-not-compliant-adfs-error/

  Thus guy found a malformed date in SAML message to be the cause of same issue


JUNO SSO Tips : Saving SSO Passwd in browser cache
-----------------------------------------------------

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






NVIDIA webpage informs me
---------------------------

Your browser blocks 3rd party cookies by default. Click on this page to allow
3rd party cookies from our partner AdRoll to help tailor ads you see. Learn
about 3rd party ads or opt out of AdRoll cookies by clicking here. This message
only appears once.

Could this setting be related to SSO difficulties



CERN Single Sign-On using Google account
-------------------------------------------

Attempting to login on https://login.cern.ch using Google account yields error

Sign in with a CERN account, a Federation account or a public service account

error   
There was a problem accessing the site.

Microsoft.IdentityServer.Web.RequestFailedException: MSIS7000: 
The sign in request is not compliant to the WS-Federation language 
for web browser clients or the SAML 2.0 protocol WebSSO profile.

Reference number: b36a9766-9b4a-48e3-bd77-e7f7a2654579


https://login.cern.ch/adfs/ls/


https://opensts.cern.ch/multists/?wa=wsignin1.0&wtrealm=https%3a%2f%2fcern.ch%2flogin&wctx=b4bd3ed5-4b9a-406e-b8cd-c9c3f9c51684




* https://social.msdn.microsoft.com/Forums/vstudio/en-US/af0ac0c0-fdc8-42aa-91f5-945a29eec333/adfs-20-web-sso-not-working-in-current-versions-of-safari-for-windows-or-ios?forum=Geneva





EOU
}
sso-dir(){ echo /tmp/$USER/env/sso ; }
sso-cd(){  mkdir -p $(sso-dir) && cd $(sso-dir); }


sso-un(){ echo $SSO_UN ; }
sso-pw(){ echo $SSO_PW ; }


sso-open(){ open $(sso-url) ; }
sso-url(){ echo http://juno.ihep.ac.cn/Dev_DocDB/0020/002046/001/llr_tutorial.htm ; }
sso-aurl(){ echo https://idp.ihep.ac.cn:443/idp/Authn/UserPassword ; }

sso-cook(){ echo cookies.txt ; }


sso-cmd-(){ cat << EOC

curl \
       -X POST \
       -F "j_username=$(sso-un)" \
       -F "j_password=$(sso-pw)" \
       --cookie $(sso-cook) \
       --cookie-jar $(sso-cook) \
       -L \
       -v $(sso-aurl) 

EOC
}

sso--(){

   sso-cd

   sso-cmd-
   sso-cmd- | sh 

}
