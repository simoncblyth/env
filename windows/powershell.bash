# === func-gen- : windows/powershell fgp windows/powershell.bash fgn powershell fgh windows
powershell-src(){      echo windows/powershell.bash ; }
powershell-source(){   echo ${BASH_SOURCE:-$(env-home)/$(powershell-src)} ; }
powershell-vi(){       vi $(powershell-source) ; }
powershell-env(){      elocal- ; }
powershell-usage(){ cat << EOU

Windows Powershell
=======================


Refs
------

* https://www.simple-talk.com/dotnet/.net-tools/further-down-the-rabbit-hole-powershell-modules-and-encapsulation/

Text File Encodings
--------------------

Default writing of files in powershell uses unexpected encodings that cause
mercurial to see the files as binary::

    PS> echo hello > hello.txt
    GB> file hello.txt 
          Little-endian UTF-16 Unicode text, with CR line terminators

    PS> echo hello | out-file -encoding ascii   ascii.txt
    PS> echo hello | out-file -encoding default default.txt
    PS> echo hello | out-file                   unspecified.txt 

          # unspecified produces "Little-endian UTF-16 Unicode text

Trying the out-file encoding options in powershell produces::

    ntuhep@ntuhep-PC MINGW64 ~/tmp
    $ file *
    ascii.txt:            ASCII text, with CRLF line terminators
    bigendianunicode.txt: Big-endian UTF-16 Unicode text, with CRLF line terminators
    default.txt:          ASCII text, with CRLF line terminators
    unspecified.txt:      Little-endian UTF-16 Unicode text, with CR line terminators
    utf8.txt:             UTF-8 Unicode (with BOM) text, with CRLF line terminators


* **CONCLUSION : SPECIFY ENCODING ASCII**



vim is able to edit the "Little-endian UTF-16 Unicode text, with CRLF, CR line terminators"
but they show up in bitbucket web interface as binary which gets downloaded when you attempt
to view.


::

    simon:psm1 blyth$ find . -name '*.psm1' -exec xxd -l 16 {} \;

    0000000: fffe 6600 7500 6e00 6300 7400 6900 6f00  ..f.u.n.c.t.i.o.
    0000000: fffe 6600 7500 6e00 6300 7400 6900 6f00  ..f.u.n.c.t.i.o.
    0000000: fffe 6600 7500 6e00 6300 7400 6900 6f00  ..f.u.n.c.t.i.o.
    0000000: fffe 6600 7500 6e00 6300 7400 6900 6f00  ..f.u.n.c.t.i.o.

    0000000: 6675 6e63 7469 6f6e 2067 6c6f 6261 6c3a  function global:
    0000000: 496d 706f 7274 2d4d 6f64 756c 6520 6f70  Import-Module op
    0000000: 6675 6e63 7469 6f6e 2065 2d73 7263 7b20  function e-src{ 
    0000000: 6675 6e63 7469 6f6e 2067 342d 7372 6320  function g4-src 
    0000000: 6675 6e63 7469 6f6e 206f 7074 6963 6b73  function opticks
    0000000: 6675 6e63 7469 6f6e 2070 732d 7372 6320  function ps-src 
    0000000: 6675 6e63 7469 6f6e 2076 732d 7372 6320  function vs-src 
    0000000: 496d 706f 7274 2d4d 6f64 756c 6520 6469  Import-Module di

    simon:psm1 blyth$ find . -name '*.psm1' -exec file {} \;
    ./assimp/assimp.psm1:   Little-endian UTF-16 Unicode text, with CRLF, CR line terminators
    ./boost/boost.psm1:     Little-endian UTF-16 Unicode text, with CRLF, CR line terminators
    ./cmak/cmak.psm1:       Little-endian UTF-16 Unicode text, with CRLF, CR line terminators
    ./glew/glew.psm1:       Little-endian UTF-16 Unicode text, with CRLF, CR line terminators

    ./clui/clui.psm1:       ASCII English text, with CRLF line terminators
    ./dist/dist.psm1:       ASCII text, with CRLF line terminators
    ./e/e.psm1:             ASCII text, with CRLF line terminators
    ./g4/g4.psm1:           ASCII English text, with CRLF line terminators
    ./opticks/opticks.psm1: ASCII text, with CRLF line terminators
    ./ps/ps.psm1:           ASCII C++ program text
    ./vs/vs.psm1:           ASCII text, with CRLF line terminators
    ./xercesc/xercesc.psm1: ASCII English text, with CRLF line terminators


* http://unix.stackexchange.com/questions/140175/normal-looking-text-file-detected-by-file-as-ascii-pascal-program-text

  Seems the language "English" "C++" detection is just some heuristic word frequency based determination.


* http://stackoverflow.com/questions/778069/how-can-i-change-a-files-encoding-with-vim


convert encodings
~~~~~~~~~~~~~~~~~~

::

    simon:boost blyth$ iconv -f utf-16 -t utf-8 boost.psm1 > c.boost.psm1
    simon:boost blyth$ file *
    boost.psm1:   Little-endian UTF-16 Unicode text, with CRLF, CR line terminators
    c.boost.psm1: ASCII text, with CRLF line terminators
    simon:boost blyth$ rm boost.psm1
    simon:boost blyth$ mv c.boost.psm1 boost.psm1


After that::

    simon:psm1 blyth$ find . -name '*.psm1' -exec file {} \;
    ./assimp/assimp.psm1: ASCII text, with CRLF line terminators
    ./boost/boost.psm1: ASCII text, with CRLF line terminators
    ./clui/clui.psm1: ASCII English text, with CRLF line terminators
    ./cmak/cmak.psm1: ASCII text, with CRLF line terminators
    ./dist/dist.psm1: ASCII text, with CRLF line terminators
    ./e/e.psm1: ASCII text, with CRLF line terminators
    ./g4/g4.psm1: ASCII English text, with CRLF line terminators
    ./glew/glew.psm1: ASCII text, with CRLF line terminators
    ./opticks/opticks.psm1: ASCII text, with CRLF line terminators
    ./ps/ps.psm1: ASCII C++ program text
    ./vs/vs.psm1: ASCII text, with CRLF line terminators
    ./xercesc/xercesc.psm1: ASCII English text, with CRLF line terminators

    simon:psm1 blyth$ find . -name '*.psm1' -exec xxd -l 16 {} \;
    0000000: 6675 6e63 7469 6f6e 2061 7373 696d 702d  function assimp-
    0000000: 6675 6e63 7469 6f6e 2062 6f6f 7374 2d73  function boost-s
    0000000: 6675 6e63 7469 6f6e 2067 6c6f 6261 6c3a  function global:
    0000000: 6675 6e63 7469 6f6e 2063 6d61 6b2d 7372  function cmak-sr
    0000000: 496d 706f 7274 2d4d 6f64 756c 6520 6f70  Import-Module op
    0000000: 6675 6e63 7469 6f6e 2065 2d73 7263 7b20  function e-src{ 
    0000000: 6675 6e63 7469 6f6e 2067 342d 7372 6320  function g4-src 
    0000000: 6675 6e63 7469 6f6e 2067 6c65 772d 7372  function glew-sr
    0000000: 6675 6e63 7469 6f6e 206f 7074 6963 6b73  function opticks
    0000000: 6675 6e63 7469 6f6e 2070 732d 7372 6320  function ps-src 
    0000000: 6675 6e63 7469 6f6e 2076 732d 7372 6320  function vs-src 
    0000000: 496d 706f 7274 2d4d 6f64 756c 6520 6469  Import-Module di
    simon:psm1 blyth$ 




PsGet : shared modules
------------------------

* http://psget.net/  
* https://github.com/psget/psget/
* https://github.com/chaliy/psurl/blob/master/PsUrl/PsUrl.psm1
* https://github.com/ligershark/psbuild

ps1 script examples
---------------------

Pre-requisite installer

* https://github.com/gadgetron/gadgetron/blob/master/doc/windows_installation/GadgetronWindowsInstallation.ps1

Choco install

* https://chocolatey.org/install.ps1

Windows copy/paste
---------------------

* select then copy: ctrl-c
* paste : shift-insert

Powershell copy/paste
------------------------

* select the text, then right-click to copy to clipboard
* paste clipboard with another right click (when into vim window get into insert mode first)



Update ?
-----------

There are choco pkgs powershell and powershell4 for 
windows management framework 5.0 and 4.0.
But maybe should try the v2 that comes with Windows7 SP1 first 

* https://chocolatey.org/packages/PowerShell
* https://chocolatey.org/packages/powershell4

Allow running scripts
----------------------

::

   Set-ExecutionPolicy Unrestricted -Scope CurrentUser

   get-host  # find version of powershell


Hello World Powershell
-----------------------

::

    PS C:\Users\ntuhep\env\windows\powershell> type hello.ps1
    Write-Host "Hello World"

    PS C:\Users\ntuhep\env\windows\powershell> & .\hello.ps1
    Hello World


Create Profile
-------------------

::

    New-Item -path $profile -type file -force

    vim $profile


Recursive rmdir
-------------------

::

    Remove-Item -Recurse -Force some-dir

    cmd /c "rd /s /q some-dir"








EOU
}
powershell-dir(){ echo $(local-base)/env/windows/windows-powershell ; }
powershell-cd(){  cd $(powershell-dir); }
powershell-mate(){ mate $(powershell-dir) ; }
powershell-get(){
   local dir=$(dirname $(powershell-dir)) &&  mkdir -p $dir && cd $dir

}
