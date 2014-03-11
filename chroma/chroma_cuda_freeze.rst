Chroma CUDA Freeze
====================

Symptoms March 10, 2014
-------------------------

::

    Freshly updated OSX: 10.9.2
    CUDA Driver Version: 5.5.28      (one before the latest)
    GPU Driver Version: 8.24.9 310.40.25f01

While doing heavy CUDA ray tracering get GUI freeze 
followed by system crash.

* http://www.mathworks.com/matlabcentral/answers/120578
* screen fills::

   cuMemFree failed: launch timeout
   PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
   cuMemFree failed: launch timeout
   PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
  
   ... message suggesting use of Context.pop() to avoid issue

* sometimes cursor reponds to input, sometime not, but nevertheless unable to do anything
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


Rare Catch
-----------

Normally dont get to see the start of the error, but somehow (potentially different conditions)::

    INFO:chroma:MOUSEBUTTONDOWN 4 [ 330.98363295  330.98363295  330.98363295] 
    INFO:chroma:MOUSEBUTTONUP 4 
    INFO:chroma:MOUSEBUTTONDOWN 4 [ 330.98363295  330.98363295  330.98363295] 
    Traceback (most recent call last):
      File "./simplecamera.py", line 285, in <module>
        camera._run()
      File "./simplecamera.py", line 271, in _run
        self.process_event(event)
      File "./simplecamera.py", line 136, in process_event
        self.translate(v)
      File "./simplecamera.py", line 119, in translate
        self.update()
      File "./simplecamera.py", line 125, in update
        pixels = self.pixels_gpu.get()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gpuarray.py", line 252, in get
        drv.memcpy_dtoh(ary, self.gpudata)
    pycuda._driver.LaunchError: cuMemcpyDtoH failed: launch timeout



Examine Logs
--------------

system.log::

    Mar 10 19:27:14 delta.local python[2861]: CPSSetForegroundOperationState(): This call is deprecated and should not be called anymore.
    Mar 10 19:27:56 delta.local python[2886]: CPSGetCurrentProcess(): This call is deprecated and should not be called anymore.
    Mar 10 19:27:56 delta.local python[2886]: CPSSetForegroundOperationState(): This call is deprecated and should not be called anymore.
    Mar 10 19:37:41 delta.local ReportCrash[2969]: Saved crash report for python[2886] version 0 to /Users/blyth/Library/Logs/DiagnosticReports/python_2014-03-10-193741_delta.crash
    Mar 10 19:37:58 delta kernel[0]: process WindowServer[108] caught causing excessive wakeups. Observed wakeups rate (per sec): 570; Maximum permitted wakeups rate (per sec): 150; Observation period: 300 seconds; Task lifetime number of wakeups: 381526
    Mar 10 19:37:58 delta.local hidd[84]: IOHIDEventQueue unable to get policy for event of type 11. (e00002e8)
    Mar 10 19:37:58 --- last message repeated 87 times ---
    Mar 10 19:37:58 delta.local ReportCrash[2972]: Invoking spindump for pid=108 wakeups_rate=570 duration=79 because of excessive wakeups
    Mar 10 19:38:06 delta.local sshd[2975]: Accepted publickey for blyth from fe80::214:51ff:fe82:2425%en0 port 58074 ssh2
    Mar 10 19:38:06 delta.local sshd: blyth [priv][2975]: USER_PROCESS: 2977 ttys003
    Mar 10 19:38:55 delta kernel[0]: NVDA(OpenGL): Channel timeout!
    Mar 10 19:39:16 delta kernel[0]: process Terminal[163] thread 1624 caught burning CPU! It used more than 50% CPU (Actual recent usage: 93%) over 180 seconds. thread lifetime cpu usage 91.563823 seconds, (52.291935 user, 39.271888 system) ledger info: balance: 90007955052 credit: 91555275475 debit: 1547320423 limit: 90000000000 (50%) period: 180000000000 time since last refill (ns): 95903014198 
    Mar 10 19:39:16 delta.local ReportCrash[3242]: Invoking spindump for pid=163 thread=1624 percent_cpu=93 duration=97 because of excessive cpu utilization
    Mar 10 19:39:23 delta kernel[0]: NVDA(OpenGL): Channel timeout!
    Mar 10 19:40:20 --- last message repeated 1 time ---
    Mar 10 19:40:20 delta kernel[0]: NVDA(OpenGL): Channel timeout!
    Mar 10 19:41:17 --- last message repeated 1 time ---
    Mar 10 19:41:17 delta kernel[0]: NVDA(OpenGL): Channel timeout!
    Mar 10 19:42:14 --- last message repeated 1 time ---
    Mar 10 19:42:14 delta kernel[0]: NVDA(OpenGL): Channel timeout!
    Mar 10 19:43:10 --- last message repeated 1 time ---
    Mar 10 19:43:10 delta kernel[0]: NVDA(OpenGL): Channel timeout!


kernel panic log::

    Anonymous UUID:       32BCAB7F-2AEA-A951-3785-013ECFB913EA

    Mon Mar 10 19:47:26 2014
    panic(cpu 0 caller 0xffffff7f9ba70fb0): "GPU Panic: [<None>] 5 3 5f 1d 0 8 0 3 : NVRM[0/1:0:0]: Read Error 0x00000144: CFG 0x0fe910de 0x00100406 0xc0000000, BAR0 0x101000000 0xffffff8200014000 0x0e7290a2, D0, P0/4\n"@/SourceCache/AppleGraphicsControl/AppleGraphicsControl-3.4.35/src/AppleMuxControl/kext/GPUPanic.cpp:127
    Backtrace (CPU 0), Frame : Return Address
    0xffffff81db8fabe0 : 0xffffff8019622fa9 
    0xffffff81db8fac60 : 0xffffff7f9ba70fb0 
    0xffffff81db8fad30 : 0xffffff7f99ffb22c 
    0xffffff81db8fadf0 : 0xffffff7f9a0c5106 
    0xffffff81db8fae30 : 0xffffff7f9a2e8e54 
    0xffffff81db8fae50 : 0xffffff7f9a001b92 
    0xffffff81db8faef0 : 0xffffff7f99fb1330 
    0xffffff81db8faf10 : 0xffffff8019ab033b 
    0xffffff81db8faf40 : 0xffffff7f99cd3a27 
    0xffffff81db8faf50 : 0xffffff7f9bb079af 
    0xffffff81db8faf60 : 0xffffff7f9bb0f7d8 
    0xffffff81db8faf80 : 0xffffff80196db3fb 
    0xffffff81db8fafd0 : 0xffffff80196f34e9 
    0xffffff81ed5abd30 : 0xffffff7f9a301d0b 
    0xffffff81ed5abda0 : 0xffffff7f9a13bdb1 
    0xffffff81ed5abe20 : 0xffffff7f99fe3aea 
    0xffffff81ed5abee0 : 0xffffff7f99fb1635 
    0xffffff81ed5abf20 : 0xffffff801964a23a 
    0xffffff81ed5abfb0 : 0xffffff80196d6ff7 
          Kernel Extensions in backtrace:
             com.apple.iokit.IOPCIFamily(2.9)[EDA75271-4E9D-34E7-A2C5-14F0C8817D37]@0xffffff7f99cbb000->0xffffff7f99ce5fff
             com.apple.driver.AppleACPIPlatform(2.0)[5D6DA288-1289-3FD4-BFA5-09C109930AE7]@0xffffff7f9bb01000->0xffffff7f9bb59fff
                dependency: com.apple.iokit.IOACPIFamily(1.4)[045D5D6F-AD1E-36DB-A249-A346E2B48E54]@0xffffff7f9a8de000
                dependency: com.apple.iokit.IOPCIFamily(2.9)[EDA75271-4E9D-34E7-A2C5-14F0C8817D37]@0xffffff7f99cbb000
             com.apple.driver.AppleMuxControl(3.4.35)[1BFF66C1-65E4-3BB3-9DEE-B61C3137019B]@0xffffff7f9ba63000->0xffffff7f9ba75fff
                dependency: com.apple.driver.AppleGraphicsControl(3.4.35)[09897896-ACBD-36B5-B1D4-0CCC4000E3B3]@0xffffff7f9ba5b000
                dependency: com.apple.iokit.IOACPIFamily(1.4)[045D5D6F-AD1E-36DB-A249-A346E2B48E54]@0xffffff7f9a8de000
                dependency: com.apple.iokit.IOPCIFamily(2.9)[EDA75271-4E9D-34E7-A2C5-14F0C8817D37]@0xffffff7f99cbb000
                dependency: com.apple.iokit.IOGraphicsFamily(2.4.1)[4421462D-2B1F-3540-8EEA-9DFCB0565E39]@0xffffff7f99f58000
                dependency: com.apple.driver.AppleBacklightExpert(1.0.4)[E04639C5-D734-3AB3-A682-FE66694C6653]@0xffffff7f9ba5e000
             com.apple.nvidia.driver.NVDAResman(8.2.4)[3D591202-DD24-3441-925A-F6808ABDF185]@0xffffff7f99fab000->0xffffff7f9a20ffff
                dependency: com.apple.iokit.IOPCIFamily(2.9)[EDA75271-4E9D-34E7-A2C5-14F0C8817D37]@0xffffff7f99cbb000
                dependency: com.apple.iokit.IONDRVSupport(2.4.1)[999E29DA-D513-3544-89D1-9885B728A098]@0xffffff7f99f9b000
                dependency: com.apple.iokit.IOGraphicsFamily(2.4.1)[4421462D-2B1F-3540-8EEA-9DFCB0565E39]@0xffffff7f99f58000
             com.apple.nvidia.driver.NVDAGK100Hal(8.2.4)[ACFCEA0C-4C80-36C0-8636-D10EE7D2DE17]@0xffffff7f9a21a000->0xffffff7f9a3c6fff
                dependency: com.apple.nvidia.driver.NVDAResman(8.2.4)[3D591202-DD24-3441-925A-F6808ABDF185]@0xffffff7f99fab000
                dependency: com.apple.iokit.IOPCIFamily(2.9)[EDA75271-4E9D-34E7-A2C5-14F0C8817D37]@0xffffff7f99cbb000

    BSD process name corresponding to current thread: kernel_task

    Mac OS version:
    13C64

    Kernel version:
    Darwin Kernel Version 13.1.0: Thu Jan 16 19:40:37 PST 2014; root:xnu-2422.90.20~2/RELEASE_X86_64
    Kernel UUID: 9FEA8EDC-B629-3ED2-A1A3-6521A1885953
    Kernel slide:     0x0000000019400000
    Kernel text base: 0xffffff8019600000
    System model name: MacBookPro11,3 (Mac-2BD1B31983FE1663)

    System uptime in nanoseconds: 6796404552470
    last loaded kext at 243252292556: com.apple.filesystems.msdosfs 1.9 (addr 0xffffff7f9bb8e000, size 65536)
    last unloaded kext at 303464511146: com.apple.filesystems.msdosfs   1.9 (addr 0xffffff7f9bb8e000, size 57344)
    loaded kexts:
    com.nvidia.CUDA 1.1.0
    com.apple.driver.AppleHWSensor  1.9.5d0
    com.apple.filesystems.autofs    3.0
    com.apple.driver.AudioAUUC  1.60
    com.apple.iokit.IOBluetoothSerialManager    4.2.3f10
    com.apple.driver.ApplePlatformEnabler   2.0.9d1
    com.apple.driver.AGPM   100.14.15
    com.apple.driver.X86PlatformShim    1.0.0
    com.apple.iokit.IOUserEthernet  1.0.0d1
    com.apple.Dont_Steal_Mac_OS_X   7.0.0
    com.apple.driver.AppleHWAccess  1
    com.apple.driver.AppleHDA   2.6.0f1
    com.apple.driver.AppleUpstreamUserClient    3.5.13
    com.apple.GeForce   8.2.4
    com.apple.iokit.BroadcomBluetoothHostControllerUSBTransport 4.2.3f10
    com.apple.driver.AppleIntelHD5000Graphics   8.2.4
    com.apple.driver.AppleSMCLMU    2.0.4d1
    com.apple.driver.AppleMCCSControl   1.1.12
    com.apple.driver.AppleMuxControl    3.4.35
    com.apple.driver.AppleLPC   1.7.0
    com.apple.driver.AppleIntelFramebufferAzul  8.2.4
    com.apple.driver.AppleThunderboltIP 1.1.2
    com.apple.driver.AppleCameraInterface   4.26.0
    com.apple.driver.AppleUSBTCButtons  240.2
    com.apple.driver.AppleUSBTCKeyboard 240.2
    com.apple.driver.AppleUSBCardReader 3.4.1
    com.apple.AppleFSCompression.AppleFSCompressionTypeDataless 1.0.0d1
    com.apple.AppleFSCompression.AppleFSCompressionTypeZlib 1.0.0d1
    com.apple.BootCache 35
    com.apple.driver.AppleUSBHub    666.4.0
    com.apple.driver.XsanFilter 404
    com.apple.iokit.IOAHCIBlockStorage  2.5.1
    com.apple.driver.AppleAHCIPort  3.0.0
    com.apple.driver.AirPort.Brcm4360   831.21.63
    com.apple.driver.AppleUSBXHCI   670.4.0
    com.apple.driver.AppleSmartBatteryManager   161.0.0
    com.apple.driver.AppleACPIButtons   2.0
    com.apple.driver.AppleRTC   2.0
    com.apple.driver.AppleHPET  1.8
    com.apple.driver.AppleSMBIOS    2.1
    com.apple.driver.AppleACPIEC    2.0
    com.apple.driver.AppleAPIC  1.7
    com.apple.nke.applicationfirewall   153
    com.apple.security.quarantine   3
    com.apple.kext.triggers 1.0
    com.apple.iokit.IOSerialFamily  10.0.7
    com.apple.iokit.IOBluetoothFamily   4.2.3f10
    com.apple.driver.DspFuncLib 2.6.0f1
    com.apple.vecLib.kext   1.0.0
    com.apple.iokit.IOAudioFamily   1.9.5fc2
    com.apple.kext.OSvKernDSPLib    1.14
    com.apple.iokit.IOAcceleratorFamily 98.14
    com.apple.nvidia.driver.NVDAGK100Hal    8.2.4
    com.apple.nvidia.driver.NVDAResman  8.2.4
    com.apple.iokit.IOBluetoothHostControllerUSBTransport   4.2.3f10
    com.apple.iokit.IOSurface   91
    com.apple.driver.X86PlatformPlugin  1.0.0
    com.apple.driver.AppleSMC   3.1.8
    com.apple.driver.AppleSMBusController   1.0.11d1
    com.apple.driver.AppleBacklightExpert   1.0.4
    com.apple.driver.AppleGraphicsControl   3.4.35
    com.apple.iokit.IONDRVSupport   2.4.1
    com.apple.driver.IOPlatformPluginFamily 5.7.0d10
    com.apple.AppleGraphicsDeviceControl    3.4.35
    com.apple.iokit.IOAcceleratorFamily2    98.14
    com.apple.driver.AppleHDAController 2.6.0f1
    com.apple.iokit.IOGraphicsFamily    2.4.1
    com.apple.iokit.IOHDAFamily 2.6.0f1
    com.apple.driver.AppleUSBMultitouch 240.9
    com.apple.iokit.IOUSBHIDDriver  660.4.0
    com.apple.driver.AppleThunderboltDPInAdapter    3.1.7
    com.apple.driver.AppleThunderboltDPAdapterFamily    3.1.7
    com.apple.driver.AppleThunderboltPCIDownAdapter 1.4.5
    com.apple.iokit.IOSCSIBlockCommandsDevice   3.6.6
    com.apple.iokit.IOUSBMassStorageClass   3.6.0
    com.apple.iokit.IOSCSIArchitectureModelFamily   3.6.6
    com.apple.driver.AppleUSBMergeNub   650.4.0
    com.apple.driver.AppleUSBComposite  656.4.1
    com.apple.iokit.IOUSBUserClient 660.4.2
    com.apple.iokit.IOAHCIFamily    2.6.5
    com.apple.driver.AppleThunderboltNHI    2.0.1
    com.apple.iokit.IOThunderboltFamily 3.2.7
    com.apple.iokit.IO80211Family   630.35
    com.apple.driver.mDNSOffloadUserClient  1.0.1b5
    com.apple.iokit.IONetworkingFamily  3.2
    com.apple.iokit.IOUSBFamily 675.4.0
    com.apple.driver.AppleEFINVRAM  2.0
    com.apple.driver.AppleEFIRuntime    2.0
    com.apple.iokit.IOHIDFamily 2.0.0
    com.apple.iokit.IOSMBusFamily   1.1
    com.apple.security.sandbox  278.11
    com.apple.kext.AppleMatch   1.0.0d1
    com.apple.security.TMSafetyNet  7
    com.apple.driver.AppleKeyStore  2
    com.apple.driver.DiskImages 371.1
    com.apple.iokit.IOStorageFamily 1.9
    com.apple.iokit.IOReportFamily  23
    com.apple.driver.AppleFDEKeyStore   28.30
    com.apple.driver.AppleACPIPlatform  2.0
    com.apple.iokit.IOPCIFamily 2.9
    com.apple.iokit.IOACPIFamily    1.4
    com.apple.kec.corecrypto    1.0
    com.apple.kec.pthread   1
     


search
~~~~~~~~

* :google:`gpu panic macbook pro retina`
  
  * https://discussions.apple.com/thread/5300005?start=15&tstart=0




CUDA Drivers 
-------------

Update History
~~~~~~~~~~~~~~~~

* http://www.nvidia.com/object/mac-driver-archive.html

::

    CUDA Mac Driver
    Latest Version: CUDA 5.5.47 driver for MAC
    Release Date: 03/05/2014

    Previous Releases: 
    CUDA 5.5.28 driver for MAC
    Release Date: 10/23/2013 

New Release 5.5.47
~~~~~~~~~~~~~~~~~~

* http://www.nvidia.com/object/macosx-cuda-5.5.47-driver.html

CUDA driver 5.5.47 is required for CUDA support on Mac OS X 10.9 Mavericks.
Please see supported products tab.

An alternative method to download the latest CUDA driver is within Mac OS
environment.  Access the latest driver through System Preferences > Other >
CUDA.  Click 'Install CUDA Update'.

Search
~~~~~~~

* http://www.toolfarm.com/blog/entry/alert_nvidia_cuda_5.5.47_update_fixes_crashes_for_several_apps_after_effect




