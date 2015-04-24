# === func-gen- : tools/mono/mono fgp tools/mono/mono.bash fgn mono fgh tools/mono
mono-src(){      echo tools/mono/mono.bash ; }
mono-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mono-src)} ; }
mono-vi(){       vi $(mono-source) ; }
mono-env(){      elocal- ; }
mono-usage(){ cat << EOU


Mono open source ECMA CLI, C# and .NET implementation. 

* https://github.com/mono/mono
* http://www.mono-project.com
* http://www.mono-project.com/download/
* http://www.monodevelop.com


Installed Mono release is: 3.12.1 with Safari and PKG style installation 

http://www.mono-project.com/docs/about-mono/supported-platforms/osx/

delta:reps blyth$ du -hs /Library/Frameworks/Mono.framework
625M    /Library/Frameworks/Mono.framework



* https://powertools.codeplex.com


::

    delta:~ blyth$ which mcs
    /usr/bin/mcs
    delta:~ blyth$ mcs --help
    Mono C# compiler, Copyright 2001-2011 Novell, Inc., Copyright 2011-2012 Xamarin, Inc
    mcs [options] source-files
       --about              About the Mono C# compiler


::

    delta:mono blyth$ ./hello.exe 
    -bash: ./hello.exe: cannot execute binary file
    delta:mono blyth$ which mono
    /usr/bin/mono
    delta:mono blyth$ mono hello.exe
    Hello Mono World
    delta:mono blyth$ mcs hello.cs -out:/tmp/hello.exe
    delta:mono blyth$ mono /tmp/hello.exe 
    Hello Mono World




EOU
}
mono-dir(){ echo $(env-home)/tools/mono ; }
mono-cd(){  cd $(mono-dir); }

