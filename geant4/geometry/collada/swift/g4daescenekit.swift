#!/usr/bin/env DEVELOPER_DIR=/Applications/Xcode6-Beta3.app/Contents/Developer xcrun swift -i
/*

G4DAESCENEKIT
===============

A Swift script that uses SceneKit to parse G4DAE exported Geant4 
geometries and dumps some characteristics of nodes. Usage::

   export-;export-export   # define the envvar
   g4daescenekit.swift

   OR

   g4daescenekit.sh 


Issues
-------

Extra element warnings
~~~~~~~~~~~~~~~~~~~~~~~~~

Many thousands of warnings from all elements within **extra** ones::

  ColladaDOM Warning: The DOM was unable to create an element named meta at line 149695. Probably a schema violation.

TODO: try to avoid this by using a different namespace for these extras

::

    (chroma_env)delta:DayaBay_VGDX_20140414-1300 blyth$ xsltproc ../strip-extra-meta.xsl g4_00.dae > g4_00.dae.noextra.dae
       
* observe that scenekit doesnt read files than do not end in .dae


Missing lots of nodes ? unique identifier problem ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Getting 5892 when expect 12k, 

* not related to the extra element warnings, as after stripping the extras get no change

* almost certainly this is the node identity problem than met 
  before with *g4daenode.py* which was fixed by ensuring that all nodes 
  have unique identifiers


FIXED undefined envvar crashes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Twas just result of undefined envvar DAE_NAME_DYB

::

    delta:~ blyth$ g4daescenekit.swift
    0  swift                    0x0000000108cb9de8 llvm::sys::PrintStackTrace(__sFILE*) + 40
    1  swift                    0x0000000108cba2d4 SignalHandler(int) + 452
    2  libsystem_platform.dylib 0x00007fff9278b5aa _sigtramp + 26
    3  libsystem_pthread.dylib  0x00007fff89fc44a9 __mtx_droplock + 491
    4  swift                    0x000000010832e219 llvm::JIT::runFunction(llvm::Function*, std::__1::vector<llvm::GenericValue, std::__1::allocator<llvm::GenericValue> > const&) + 329
    5  swift                    0x00000001085da533 llvm::ExecutionEngine::runFunctionAsMain(llvm::Function*, std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > const&, char const* const*) + 1523
    6  swift                    0x00000001082174da swift::RunImmediately(swift::CompilerInstance&, std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > const&, swift::IRGenOptions&, swift::SILOptions const&) + 1066
    7  swift                    0x000000010804dcd3 frontend_main(llvm::ArrayRef<char const*>, char const*, void*) + 5283
    8  swift                    0x000000010804c81d main + 1533
    9  libdyld.dylib            0x00007fff8dd985fd start + 1
    10 libdyld.dylib            0x000000000000000c start + 1915124240
    Stack dump:
    0.  Program arguments: /Applications/Xcode6-Beta3.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift -frontend -i /Users/blyth/env/bin/g4daescenekit.swift -enable-objc-attr-requires-objc-module -target x86_64-apple-darwin13.3.0 -module-name g4daescenekit -sdk /Applications/Xcode6-Beta3.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk -color-diagnostics 
    Illegal instruction: 4
    delta:~ blyth$ 


Commandline args need to be after double dash 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variability makes commandline args difficult to use, so 
will need to place scripts in shell wrappers to manage the commandline::

    delta:~ blyth$ g4daescenekit.swift 
    index 0 arg -i 
    index 1 arg /Users/blyth/env/bin/g4daescenekit.swift 
    index 2 arg -enable-objc-attr-requires-objc-module 
    index 3 arg -target 
    index 4 arg x86_64-apple-darwin13.3.0 
    index 5 arg -module-name 
    index 6 arg g4daescenekit 
    index 7 arg -sdk 
    index 8 arg /Applications/Xcode6-Beta3.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk 
    index 9 arg -color-diagnostics 
    envvar DAE_NAME_DYB is not defined 


    delta:~ blyth$ g4daescenekit.swift hello
    <unknown>:0: error: no such file or directory: 'hello'

    delta:~ blyth$ g4daescenekit.swift -- hello
    index 0 arg hello 
    envvar DAE_NAME_DYB is not defined 
    delta:~ blyth$ 
    delta:~ blyth$ 



*/

import Foundation
import SceneKit 

for i in 0..<C_ARGC {
    let index = Int(i);
    let arg = String.fromCString(C_ARGV[index])
    println("index \(index) arg \(arg) ")
}

for (index,arg) in enumerate(Process.arguments) {
   println("index \(index) arg \(arg) ")
}



let env = NSProcessInfo.processInfo().environment
let key = "DAE_NAME_DYB"


if let path:String = env[key] as? NSString {

   let urlpath = path + ".noextra.dae"
   //let urlpath = path 
   println("urlpath \(urlpath) ")

   let url = NSURL(fileURLWithPath:urlpath)

   let sceneSource = SCNSceneSource(URL:url, options:nil)
   var nid = sceneSource.identifiersOfEntriesWithClass(SCNNode.self) as? [String]

   for (index, identifier) in enumerate(nid!) {
       println("ix:\(index) id:\(identifier)")
   }
   println(nid!.count)

} else {
   println("envvar \(key) is not defined ")
}


