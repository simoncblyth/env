openal-usage(){ cat << EOU


* https://www.openal.org/

* https://stackoverflow.com/questions/65492484/is-openal-open-source-where-is-the-source-code

* https://indiegamedev.net/2020/04/12/the-complete-guide-to-openal-with-c-part-3-positioning-sounds/

* https://www.g2.com/products/openal/competitors/alternatives

Wwise fmod SoLoud Miles Allegro AstoundSound 3Dception


* https://www.reddit.com/r/gamedev/comments/4wlp2n/whats_the_best_crossplatform_audio_lib_these_days/

* https://github.com/andrewrk/libsoundio

* https://en.wikipedia.org/wiki/Audiokinetic_Wwise

  Used by many games 


* https://medium.com/@akash.thakkar/should-you-learn-fmod-or-wwise-to-work-in-game-audio-936fe6295c0


* https://forums.developer.nvidia.com/t/audio-processing-via-nvidia-gpu/194862

* https://developer.nvidia.com/vrworks-audio-sdk-depth

The core technology of the NVIDIA VRWorks Audio SDK is a geometric acoustics
ray-tracing engine, called NVIDIA Acoustic Raytracer (NVAR).

NVIDIA VRWorks Audio is the only fully hardware-accelerated and path-traced
audio solution which creates a complete acoustic image of the environment in
real-time without requiring any predetermined filters. VRWorks Audio library
does not require any “pre-baked” knowledge of the scene. As the scene is loaded
by the application, the acoustic model is built and updated on-the-fly; and
audio effect filters are generated and applied in real time on the sound source
waveforms. This approach gives tremendous time-savings to the audio designers
and engineers, because it allows them to focus on designing the soundscapes
rather than thinking about how to render it well. Rendering is automatically
taken care by NVAR library. 

For example, a typical game level with a large building with multiple rooms and
architectural features will require fine tuning of audio effects in each of
these rooms. Generating these effects accurately is an iterative process and
requires several man-weeks’ effort. With VRWorks Audio, this time reduces to
zero.

* https://developer.nvidia.com/blog/vrworks-audio-dials-up-the-immersion-with-rtx-acceleration/

* ~/opticks_refs/NVIDIA_VRWorks_Audio_SDK_Overview.pdf
* https://developer.download.nvidia.com/assets/gameworks/downloads/secure/VRWorks-Audio-1.0.0/VRWorks%20Audio%20SDK%20Overview.pdf
* Windows only ? 

* https://forums.developer.nvidia.com/t/linux-support-for-vrworks-360-video-sdk/50264/6



EOU
}
