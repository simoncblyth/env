#include "ORenderer.hh"

#include <cstddef>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// brap-
#include "BTimer.hh"
#include "PLOG.hh"

// optixrap-
#include "OFrame.hh"

// oglrap-
#include "Renderer.hh"
#include "Texture.hh"


ORenderer::ORenderer(Renderer* renderer, OFrame* frame, const char* dir, const char* incl_path)
    :
    m_frame(frame),
    m_renderer(renderer),
    m_texture(NULL),
    m_texture_id(-1),

    m_render_count(0),
    m_render_prep(0),
    m_render_time(0)
{
    init(dir, incl_path);
}


void ORenderer::init(const char* /*dir*/, const char* /*incl_path*/)
{
    // TODO: move elsewhere ... diddling with another objects constituent

    Texture* texture = m_frame->getTexture();
    m_renderer->upload(texture);
}


void ORenderer::setSize(unsigned int width, unsigned int height)
{
    assert(0);

    m_frame->setSize( width, height) ;

}


void ORenderer::render()
{
    LOG(debug) << "ORenderer::render " << m_render_count ; 

    double t0 = BTimer::RealTime();

    m_frame->push_PBO_to_Texture();

    double t1 = BTimer::RealTime();

    m_renderer->render();

    double t2 = BTimer::RealTime();

    m_render_count += 1 ; 
    m_render_prep += t1 - t0 ; 
    m_render_time += t2 - t1 ; 

    glBindTexture(GL_TEXTURE_2D, 0 );  

    if(m_render_count % 10 == 0) report("ORenderer::render"); 
}



void ORenderer::report(const char* msg)
{
    if(m_render_count == 0) return ; 

    std::stringstream ss ; 
    ss 
          << " render_count    " << std::setw(10) << m_render_count  << std::endl 
          << " render_prep     " << std::setw(10) << m_render_prep  << " avg " << std::setw(10) << m_render_prep/m_render_count  << std::endl
          << " render_time     " << std::setw(10) << m_render_time  << " avg " << std::setw(10) << m_render_time/m_render_count  << std::endl
           ;

    LOG(debug) << msg ;
    LOG(debug) << ss.str() ; 

}


