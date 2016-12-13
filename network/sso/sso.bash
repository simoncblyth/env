# === func-gen- : network/sso/sso fgp network/sso/sso.bash fgn sso fgh network/sso
sso-src(){      echo network/sso/sso.bash ; }
sso-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sso-src)} ; }
sso-vi(){       vi $(sso-source) ; }
sso-env(){      elocal- ; }
sso-usage(){ cat << EOU

SSO Debugging
================

CERN SSO and IHEP SSO both giving problems from Taiwan ??

* http://myitforum.com/myitforumwp/2014/01/13/the-so-annoying-msis7000-signin-request-not-compliant-adfs-error/

  Thus guy found a malformed date in SAML message to be the cause of same issue


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
sso-dir(){ echo $(local-base)/env/network/sso/network/sso-sso ; }
sso-cd(){  cd $(sso-dir); }
sso-mate(){ mate $(sso-dir) ; }
sso-get(){
   local dir=$(dirname $(sso-dir)) &&  mkdir -p $dir && cd $dir

}
