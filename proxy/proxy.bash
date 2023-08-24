proxy-vi(){  vi $BASH_SOURCE ; }
proxy-usage(){ cat << EOU
PROXY
======


Proxy suggestion from Tao, Aug 3, 2023
-----------------------------------------

::

    Hi Simon,

    I think Tian may explain the reason. 
    Maybe you can try to pull via proxies.

    If you are using http protocol, please edit $HOME/.gitconfig:

    [http "https://github.com"]
            proxy = socks5://localhost:12345
    If using ssh protocol, please edit $HOME/.ssh/config:

    Host *github.com
        User git
        ProxyCommand nc -v --proxy 127.0.0.1:12345 --proxy-type socks5 %h %p

    Tao


socks.pac and Safari.app
-------------------------

For usage of socks.pac file with Safari.app  see :doc:`/bash/ssh`



EOU
}
