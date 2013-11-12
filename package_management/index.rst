Package Management
=====================

* http://en.wikipedia.org/wiki/List_of_software_package_management_systems

Cross platform management systems ?
-------------------------------------

* :google:`cross platform package management`


OpenPKG
--------

* http://en.wikipedia.org/wiki/OpenPKG
* http://www.openpkg.org/

Nix
-----

* http://nixos.org/
* http://nixos.org/nix/
* http://nixos.org/nix/manual/

* http://en.wikipedia.org/wiki/Nix_package_manager
* http://www.gnu.org/ghm/2009/dolstra-nix-slides.pdf

* :google:`/nix/store`
* :google:`Efficient Upgrading in a Purely Functional Component Deployment Model. Eelco Dolstra`

  * http://nixos.org/~eelco/talks/eupfcdm-cbse-2005.pdf
  * http://nixos.org/~eelco/talks/
  * http://www.gnu.org/ghm/2009/


.. sidebar:: More precise dependency tracking ?
      
     At expense of many versions of packages 


Nix is a *purely functional package manager*. This means that it treats packages
like values in purely functional programming languages such as Haskell. They
are built by functions that dont have side-effects, and they never change
after they have been built. Nix stores packages in the Nix store, usually the
directory ``/nix/store``, where each package has its own unique subdirectory such
as ``/nix/store/nlc4z5y1hm8w9s8vm6m1f5hy962xjmp5-firefox-12.0`` where ``nlc4z5...`` 
is a unique identifier for the package that captures all its dependencies 
(its a cryptographic hash of the packages build dependency graph). 
This enables many powerful features.

#. Uses directories named by dependency tree hashes to keep things
   more tightly controlled. 
#. Also allows non-root installation.
#. Uses its own expression language, devised for software dependency handling 
#. should run on most Unix systems, including Linux, FreeBSD and Mac OS X.


nixpkgs
~~~~~~~~~~

* http://nixos.org/nixpkgs/manual/  
* https://github.com/NixOS/nixpkgs
* https://github.com/NixOS/nixpkgs/tree/master/pkgs/development/libraries




