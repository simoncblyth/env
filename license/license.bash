license-source(){   echo ${BASH_SOURCE} ; }
license-vi(){   vi $(license-source) ; }
license-env(){  elocal- ; }
license-usage(){ cat << EOU

LICENSE
=========

* https://snyk.io/blog/mit-apache-bsd-fairest-of-them-all/

MIT’s permissive license is used on 51% of Github projects. 
Apache 2.0, the runner-up, is used by another 14.8%, 
and 6.6% use either the 2-clause or 3-clause version of the
original BSD license. 
A 72% market share for the leading permissive licenses,
at least among Github users.


Copyright notice too ? At the head of the LICENSE

Copyright (C) 2019 Simon C Blyth and other contributors



* https://www.apache.org/licenses/LICENSE-2.0
* https://www.apache.org/licenses/LICENSE-2.0.txt


 license-;license--
      downloads the Apache 2.0 license into pwd, splitting 
      off the appendix     

EOU
}


license-project(){ echo ${LICENSE_PROJECT:-Opticks} ; }

license-year(){ date +"%Y" ; }

license-copyright-(){ cat << EOC
   Copyright (C) $(license-year) The $(license-project) Authors. All rights reserved.
EOC
}

license-apache2(){
    #curl https://www.apache.org/licenses/LICENSE-2.0.txt | head -177 > LICENSE

    [ -f LICENSE ] && echo LICENSE already in $PWD && return 

    license-copyright- > LICENSE
    curl https://www.apache.org/licenses/LICENSE-2.0.txt  >> LICENSE
}

license--(){ 
    license-apache2 
}



license-mit(){
    [ -f LICENSE ] && echo LICENSE already in $PWD && return 
    $FUNCNAME- > LICENSE
}

license-mit-(){ cat << EOM
The MIT License (MIT)

Copyright (c) $(license-year) Simon C Blyth and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
EOM
}



