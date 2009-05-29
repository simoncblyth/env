
//
// DIRECT	 Fetch the object directly from the content HTTP server denoted by its URL
// PROXY name:port	 Fetch the object via the proxy HTTP server at the given location (name and port)
// SOCKS name:port	 Fetch the object via the SOCKS server at the given location (name and port)
//
//
//
//  command line testing ... does not go thru the pac ?  
//
//    curl --socks5 localhost:8080 http://192.168.37.177:8080/tracs/dybsvn/
//
//   to test a new pac have to bounce safari 
//
//
//   http://www.microsoft.com/technet/prodtechnol/ie/ieak/techinfo/deploy/60/en/corpexjs.mspx?mfr=true


function FindProxyForURL(url, host) {
   if (url.substring(0, 14) == "http://192.168") {
      return "SOCKS 127.0.0.1:8080" ;
   }
   //if (url.substring(0,14) == "http://belle7." ){
   //   return "SOCKS 127.0.0.1:8080" ;
   //}
   if (url.substring(0,33) == "http://cms01.phys.ntu.edu.tw:9090" ){
      return "SOCKS 127.0.0.1:9090" ;
   }
   return "DIRECT" ;
}


function just_socks_works(url, host) {
   return "SOCKS 127.0.0.1:8080";   
    //  Safari succeeds with    http://192.168.37.177:8080/tracs/dybsvn/   
}


function oldFindProxyForURL(url, host) {
     // our local URLs from the domains below foo.com don't need a proxy:
     if (shExpMatch(url,"*.foo.com/*"))                  {return "DIRECT";}
     if (shExpMatch(url, "*.foo.com:*/*"))               {return "DIRECT";}
    
    
     if (shExpMatch(url, "192.168.37.177:*/*"))  { 
          return "SOCKS 127.0.0.1:8080";
      }    
    
      
    //if (isInNet(host, "192.168.0.0",  "255.255.0.0"))    {
    //    return "SOCKS5 127.0.0.1:8080";
    //}
    
           
     // URLs within this network are accessed through 
     // port 8080 on fastproxy.foo.com:
     if (isInNet(host, "10.0.0.0",  "255.255.248.0"))    {
        return "PROXY fastproxy.foo.com:8080";
     }
     
     // All other requests go through port 8080 of proxy.foo.com.
     // should that fail to respond, go directly to the WWW:
     // BUT it seems that it waits for proxy.foo.com:8080 to timeout ... 
     // return "PROXY proxy.foo.com:8080; DIRECT";
     
     return "DIRECT" ;
     
     // comment the above and try the below to cause deliberate failure
     //  ... succeeds to fails : Safari cannot load anything
     //  ... modifying the PAC file does not trigger a Safari re-conf
     //return "PROXY proxy.foo.com:8080";
     
}


