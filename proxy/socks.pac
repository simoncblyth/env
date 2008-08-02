
function FindProxyForURL(url, host) {
     // our local URLs from the domains below foo.com don't need a proxy:
     if (shExpMatch(url,"*.foo.com/*"))                  {return "DIRECT";}
     if (shExpMatch(url, "*.foo.com:*/*"))               {return "DIRECT";}
     
     // URLs within this network are accessed through 
     // port 8080 on fastproxy.foo.com:
     if (isInNet(host, "10.0.0.0",  "255.255.248.0"))    {
        return "PROXY fastproxy.foo.com:8080";
     }
     
     // All other requests go through port 8080 of proxy.foo.com.
     // should that fail to respond, go directly to the WWW:
     return "PROXY proxy.foo.com:8080; DIRECT";
}


