# === func-gen- : /Users/blyth/e/graphics/vrml/instant_reality_player/eai fgp /Users/blyth/e/graphics/vrml/instant_reality_player/eai.bash fgn eai fgh /Users/blyth/e/graphics/vrml/instant_reality_player
eai-src(){      echo graphics/vrml/instant_reality_player/eai.bash ; }
eai-source(){   echo ${BASH_SOURCE:-$(env-home)/$(eai-src)} ; }
eai-vi(){       vi $(eai-source) ; }
eai-env(){      elocal- ; }
eai-usage(){ cat << EOU

INSTANT REALITY PLAYER EXTERNAL AUTHORING INTERFACE EAI
===========================================================

* http://doc.instantreality.org/tutorial/external-authoring-interface-javanet/

* :google:`java VRML EAI`
* http://tecfa.unige.ch/guides/vrml/vrml97/ExternalInterface.html

* http://www.web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/
* http://doc.instantreality.org/media/apidocs/java/index.html


Is the VRML model defining the class ?
----------------------------------------

::

    Browser.WorldURL = "org.instantreality.vrml.eai.net.Node@b62aab"
    vrml.eai.field.InvalidEventInException: Node of type "Shape" does not have an EventIn "set_diffuseColor"
            at org.instantreality.vrml.eai.net.InternalNode.getEventIn(Unknown Source)
            at org.instantreality.vrml.eai.net.Node.getEventIn(Unknown Source)
            at EAIFramework.main(EAIFramework.java:25)



* http://www.cs.auckland.ac.nz/references/vrml/x3d/part2/javaBind.html
* http://graphcomp.com/info/specs/sgi/vrml/spec/part1/nodesRef.html#Shape

Shape spec::

    Shape {
      exposedField SFNode appearance NULL
      exposedField SFNode geometry   NULL
    }




Making sense of VRML
---------------------

* :google:`VRML event model`

    * 
    * http://webcache.googleusercontent.com/search?q=cache:DNy5xduDoKUJ:vrmlworks.crispen.org/tutorials/events.html+VRML+event+model&cd=1&hl=en&ct=clnk&gl=tw&client=safari
    * its worth rooting around in the cache for the above

    * http://tecfa.unige.ch/guides/vrml/vrmlman/node27.html



::

    simon:Desktop blyth$ ln -s Instant\ Player.app Instant_Player.app

::

    simon:instant_reality_player blyth$ java -cp $(eai-jar):. EAIFramework 
    Browser.Name = "Avalon"
    Browser.Version = "V2.3.0 build: R-25322 Jul 18 2013 Mac OS X ppc"
    Browser.CurrentSpeed = 1.0
    Browser.CurrentFrameRate = 45.059345
    Browser.WorldURL = "http://belle7.nuu.edu.tw/wrl/around_dupe_named.wrl"
    simon:instant_reality_player blyth$ 


EOU
}
eai-dir(){ echo $(env-home)/graphics/vrml/instant_reality_player  ; }
eai-cd(){  cd $(eai-dir); }
eai-jar(){ echo /Users/blyth/Desktop/Instant_Player.app/Contents/MacOS/instantreality.jar ; }
eai-get(){
    local names="EAIFramework.java EAIExample.wrl EAIExample.java"
    local urlbase="http://doc.instantreality.org/tutorial/external-authoring-interface-javanet"
    for name in $names ; do 
       [ ! -f $name ] && curl -L -O $urlbase/$name 
    done
}


eai-framework(){ 
   javac -cp $(eai-jar) EAIFramework.java
   java -cp $(eai-jar):. EAIFramework 
} 
eai-example(){ 
   javac -cp $(eai-jar) EAIExample.java
   java -cp $(eai-jar):. EAIExample 
} 


