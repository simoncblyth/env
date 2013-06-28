/*
JNI Hello World
=================

* http://mrjoelkemp.com/2012/01/getting-started-with-jni-and-c-on-osx-lion/

Tutorial suggests the below, but neither dirs exist on b2mc::

   -I/System/Library/Frameworks/JavaVM.framework/Versions/CurrentJDK/Headers \
   -I/Developer/SDKs/MacOSX10.6.sdk/System/Library/Frameworks/JavaVM.framework/Versions/A/Headers \

::

   javac HelloWorld.java       # .java to .class

   javah -jni HelloWorld       # .java to .h  

   g++ \
       -I/System/Library/Frameworks/JavaVM.framework/Versions/Current/Headers \
       -c HelloWorld.cpp          # .cpp to .o  linking against jni.h

   g++ -dynamiclib -o libhelloworld.jnilib HelloWorld.o

   java HelloWorld

   rm *.class *.o *.jnilib      # remove derived files


*/

class HelloWorld {
    private native void print();
    public static void main(String[] args) {
        new HelloWorld().print();
    }
    static {
        System.loadLibrary("HelloWorld");
    }
}


