api-vi(){ vi $BASH_SOURCE ; }
api-env(){ echo -n ; }
api-usage(){ cat << EOU
Comparison of Graphics API
=============================

Thinking about abstraction on top of:

1. Vulkan
2. Metal
3. DirectX11,12
4. OpenGL 


* https://alain.xyz/blog/comparison-of-modern-graphics-apis

* https://alextardif.com/RenderingAbstractionLayers.html

* https://www.gamedeveloper.com/programming/designing-a-modern-cross-platform-low-level-graphics-library


* https://media.contentapi.ea.com/content/dam/ea/seed/presentations/wihlidal-halcyonarchitecture-notes.pdf

* http://diligentgraphics.com/diligent-engine/

* https://github.com/DiligentGraphics/DiligentEngine

  * Metal backend : Available under commercial license





EOU
}
