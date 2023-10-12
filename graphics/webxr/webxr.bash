webxr-vi(){ vi $BASH_SOURCE ; }
webxr-env(){ echo -n ; }
webxr-usage(){ cat << EOU
WebXR
======

* https://immersiveweb.dev/


OpenXR vs WebXR
------------------

They do different things. WebXR from W3C (internet standards group) is for
making VR/AR apps that are accessed as a web page through the internet. OpenXR
from Khronos Group is meant to be more of a standard for native VR/AR apps that
run on the device. OpenXR is a standard for XR runtimes


Apple WebXR
-------------

* https://developer.apple.com/forums/thread/733772

VisionOS comes with a Safari that supports WebXR. You can test in on the Vision
Pro Simulator. Sadly, as you're figuring out, a version of Safari with WebXR
support will only be available for Vision Pro and not iOS devices. This is
discussed in greater length here:

* https://www.lowpass.cc/p/apple-vision-pro-webxr-standard-ios.





EOU
}


