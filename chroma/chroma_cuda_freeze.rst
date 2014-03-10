Chroma CUDA Freeze
====================

While doing heavy ray tracer activity get 


* http://www.mathworks.com/matlabcentral/answers/120578
* screen fills::

   cuMemFree failed: launch timeout
   PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
   cuMemFree failed: launch timeout
   PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
  
   ... message suggesting use of Context.pop() to avoid issue


* sometimes cursr reponds to input, sometime not, but nevertheless unable to do anything
  essentially a GUI freeze
* fans spin up, then down, then up

SSH in from another machinem,  using top observe process with name beginning DumpGPU::

    delta:~ blyth$ mdfind DumpGPU
    /System/Library/Sandbox/Profiles/com.apple.DumpGPURestart.sb
    /System/Library/LaunchDaemons/com.apple.DumpGPURestart.plist
    /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/DumpGPURestart
    delta:~ blyth$ 
    delta:~ blyth$ 


* sometimes get after 5 minutes a blank white screen and then the grey multi lingual death message::

   You computer restarted because of a problem




