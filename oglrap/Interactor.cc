#include <cstdio>
#include <string>
#include <sstream>
#include <cstring>


#include "PLOG.hh"

#include "NGLM.hpp"

#include "Opticks.hh"
#include "OpticksConst.hh"
#include "Composition.hh"
#include "Bookmarks.hh"
#include "Camera.hh"
#include "View.hh"
#include "Trackball.hh"
#include "Clipper.hh"
#include "Animator.hh"


#include "Frame.hh"
#include "Touchable.hh"
#include "Scene.hh"
#include "Interactor.hh"

//#include <GLFW/glfw3.h>   // for the key definitions maybe recode to avoid this include 


#ifdef GUI_
#include <imgui.h>
#endif

const unsigned int Interactor::_pan_mode_key = GLFW_KEY_X ; 

const char* Interactor::DRAGFACTOR = "dragfactor" ; 
const char* Interactor::OPTIXMODE  = "optixmode" ; 


Interactor::Interactor() 
   :
   m_composition(NULL),
   m_bookmarks(NULL),
   m_camera(NULL),
   m_view(NULL),
   m_trackball(NULL),
   m_clipper(NULL),
   m_touchable(NULL),
   m_frame(NULL),
   m_scene(NULL),
   m_animator(NULL),
   m_zoom_mode(false), 
   m_pan_mode(false), 
   m_near_mode(false), 
   m_far_mode(false), 
   m_yfov_mode(false),
   m_scale_mode(false),
   m_rotate_mode(false),
   m_bookmark_mode(false),
   m_gui_mode(false),
   m_scrub_mode(false),
   //m_optix_mode(0),
   m_optix_resolution_scale(1),
   m_dragfactor(1.f),
   m_container(0),
   m_changed(true),
   m_gui_style(NONE)
{
   for(unsigned int i=0 ; i < NUM_KEYS ; i++) m_keys_down[i] = false ; 
   m_status[0] = '\0' ;
}






bool* Interactor::getGUIModeAddress()
{
    return &m_gui_mode ; 
}

bool* Interactor::getScrubModeAddress()
{
    return &m_scrub_mode ; 
}

void Interactor::nextGUIStyle()
{
    int next = (m_gui_style + 1) % NUM_GUI_STYLE ; 
    m_gui_style = (GUIStyle_t)next ; 
    applyGUIStyle();
}

void Interactor::applyGUIStyle()  // G:key 
{
    switch(m_gui_style)
    {
        case NONE:
                  m_gui_mode = false ;    
                  m_scrub_mode = false ;    
                  break ; 
        case SCRUB:
                  m_gui_mode = false ;    
                  m_scrub_mode = true ;    
                  break ; 
        case FULL:
                  m_gui_mode = true ;    
                  m_scrub_mode = true ;    
                  break ; 
        default:
                  break ; 
                   
    }
}






/*
bool* Interactor::getModeAddress(const char* name)
{
    if(strcmp(name, GUI_MODE)==0) return &m_gui_mode ;
    return NULL ;
}
*/


void Interactor::setScene(Scene* scene)
{
    m_scene = scene ; 
}
void Interactor::setTouchable(Touchable* touchable)
{
    m_touchable = touchable ; 
}
Touchable* Interactor::getTouchable()
{
    return m_touchable ; 
}
void Interactor::setFrame(Frame* frame)
{
    m_frame = frame ; 
}
Frame* Interactor::getFrame()
{
    return m_frame ; 
}


void Interactor::setContainer(unsigned int container)
{
    printf("Interactor::setContainer %u \n", container);
    m_container = container ; 
}

unsigned int Interactor::getContainer()
{
    return m_container ;  
}


/*
bool Interactor::isOptiXMode()
{ 
    return m_optix_mode > 0 ; 
}
void Interactor::setOptiXMode(int optix_mode)
{
    m_optix_mode =  optix_mode ; 
    m_changed = true ; 
}
int Interactor::getOptiXMode()
{ 
    return m_optix_mode ; 
}
*/



unsigned int Interactor::getOptiXResolutionScale()
{ 
    return m_optix_resolution_scale  ; 
}
void Interactor::setOptiXResolutionScale(unsigned int scale)
{
    m_optix_resolution_scale = (scale > 0 && scale <= 32) ? scale : 1 ;  
    printf("Interactor::setOptiXResolutionScale %u \n", m_optix_resolution_scale);
    m_changed = true ; 
}


bool Interactor::hasChanged()
{
    return m_changed ; 
}
void Interactor::setChanged(bool changed)
{
    m_changed = changed ; 
}




void Interactor::nextOptiXResolutionScale(unsigned int modifiers)
{
    if(modifiers & OpticksConst::e_shift)
        setOptiXResolutionScale(getOptiXResolutionScale()/2);
    else
        setOptiXResolutionScale(getOptiXResolutionScale()*2);
}




/*
const char* Interactor::GUIMODE    = "gui" ; 
const char* Interactor::SCRUBMODE  = "scrub" ; 
*/

void Interactor::gui()
{
#ifdef GUI_
    ImGui::Text(" status: %s\n\n%s", m_status, keys  );
#endif    
}

void Interactor::configureF(const char* /*name*/, std::vector<float> /*values*/)
{
    LOG(debug)<<"Interactor::configureF";
}

void Interactor::configureI(const char* /*name*/, std::vector<int> values)
{
    LOG(debug) << "Interactor::configureI";
    if(values.empty()) return ;

    /*
    if(strcmp(name, OPTIXMODE) == 0)
    {
        int last = values.back();
        setOptiXMode(last);
    }
    */
}




void Interactor::setBookmarks(Bookmarks* bookmarks)
{
    m_bookmarks = bookmarks ;
}

void Interactor::setComposition(Composition* composition)
{
    m_composition = composition ;
    m_camera = composition->getCamera() ;
    m_view   = composition->getView();
    m_trackball = composition->getTrackball();
    m_clipper  = composition->getClipper();
    m_animator = NULL ;  // defer
}


void Interactor::cursor_drag(float x, float y, float dx, float dy, int ix, int iy )
{
    m_changed = true ; 
    //printf("Interactor::cursor_drag x,y  %0.5f,%0.5f dx,dy  %0.5f,%0.5f \n", x,y,dx,dy );

    float df = m_dragfactor ; 

    if( m_yfov_mode )
    {
        m_camera->zoom_to(df*x,df*y,df*dx,df*dy);
    }
    else if( m_near_mode )
    {
        m_camera->near_to(df*x,df*y,df*dx,df*dy);
    }
    else if( m_far_mode )
    {
        m_camera->far_to(df*x,df*y,df*dx,df*dy);
    }
    else if( m_scale_mode )
    {
        m_camera->scale_to(df*x,df*y,df*dx,df*dy);
    }
    else if( m_pan_mode )
    { 
        m_trackball->pan_to(df*x,df*y,df*dx,df*dy);
    } 
    else if( m_zoom_mode )  // bad name, actully z translate
    { 
        m_trackball->zoom_to(df*x,df*y,df*dx,df*dy);
    } 
    else if( m_rotate_mode )
    {
        m_trackball->drag_to(df*x,df*y,df*dx,df*dy);
    }
    /*
    else if( m_scrub_mode )
    {
        if(!m_animator) m_animator = m_composition->getAnimator();
        m_animator->scrub_to(df*x,df*y,df*dx,df*dy);
    }
    */
    else
    {
        m_frame->touch(ix, iy );  
        // unProjects cursor position and identifiers smallest containing volume
        // frame records volume index of what is touched 
    }
}


const char* Interactor::keys = 
"\n A: Composition::nextMode     record animation, enable and control speed  "
"\n B: Scene::nextGeometryStyle  bbox/norm/wire "
"\n C: Clipper::next             toggle geometry clipping "
"\n D: Camera::nextStyle         perspective/orthographic "
"\n E: Composition::nextGeometryStyle  lightshader/normalshader/flatvertex/facecolor "
"\n F: far mode toggle : swipe up/down change frustum far "
"\n G: gui mode    toggle GUI "
"\n H: Trackball::home  "
"\n I: Scene::nextInstanceStyle style of instanced geometry eg PMT visibility  "
"\n J: Scene::jump  "
"\n K: Composition::nextPickPhotonStyle OR toggle scrub mode "
"\n L: Composition::nextNormalStyle     flip normal in shaders "
"\n M: Composition::nextColorStyle      m1/m2/f1/f2/p1/p2      "
"\n N: near mode toggle : swipe up/down to change frustum near "  
"\n O: OptiX render mode  toggle "
"\n P: Scene::nextPhotonStyle       dot/longline/shortline  "
"\n Q: Scene::nextGlobalStyle      non-instanced geometry style: default/normalvec/none "
"\n R: rotate mode toggle  drag around rotate around viewpoint " 
"\n S: screen scale mode toggle  drag up/down to change screen scale " 
"\n T: Composition::nextViewMode " 
"\n U: Composition::nextViewType, use to cycle thru view/altview : altview is InterpolatedView  " 
"\n V: View::nextMode      rotate view, with shift modifier rotates in opposite direction "    
"\n W: decrease(increase with shift modifier) OptiX rendering resolution by multiples of 2, up to 16x"
"\n X: pan mode toggle "
"\n Y: yfov mode toggle "
"\n Z: zoom mode toggle   (actually changes z position, not zoom) " 
"\n 0-9: jump to preexisting bookmark  " 
"\n 0-9 + shift: create or update bookmark  " 
"\n SPACE: update the current bookmark, commiting trackballing into the view and persisting "
"\n "
"\n Animator modes are changed by pressing keys A,V,T"
"\n "
"\n A: event propagation "
"\n V: geometry rotation "
"\n T: interpolated navigation "
"\n "
"\n Holding SHIFT with A,V,T reverses animation time direction "
"\n Holding OPTION with A,V,T changes to previous animation mode, instead of next  "
"\n Holding CONTROL with A,V,T sets animation mode to OFF  "
"\n "
"\n ";

void Interactor::key_pressed(unsigned int key)
{
    m_changed = true ; 

    if(key < NUM_KEYS) m_keys_down[key] = true ; 

    if(key > 245) printf("Interactor::key_pressed %u \n", key );
    unsigned int modifiers = getModifiers(); 

    switch (key)
    {
        //  ABCDEFGHIJKLMNOPQRSTUVWXYZ
        //  **************************

        case GLFW_KEY_A:
            m_composition->nextAnimatorMode(modifiers) ; 
            break;
        case GLFW_KEY_B:
            m_scene->nextGeometryStyle(); 
            break;
        case GLFW_KEY_C:
            m_clipper->next(); 
            break;
        case GLFW_KEY_D:
            m_camera->nextStyle(); 
            break;
        case GLFW_KEY_E:
            m_composition->nextGeometryStyle(); 
            break;
        case GLFW_KEY_F:
            m_far_mode = !m_far_mode ; 
            break;
        case GLFW_KEY_G:
            printf("Interactor:G\n");
            //m_gui_mode = !m_gui_mode ; 
            nextGUIStyle();
            break;
        case GLFW_KEY_H:
            m_composition->home(); 
            break;
        case GLFW_KEY_I:
            m_scene->nextInstanceStyle(); 
            break;
        case GLFW_KEY_J:
            m_scene->jump(); 
            break;
        case GLFW_KEY_K:
            //m_composition->nextPickPhotonStyle(); 
            m_scrub_mode = !m_scrub_mode ; 
            break;
        case GLFW_KEY_L:
            m_composition->nextNormalStyle(); 
            break;
        case GLFW_KEY_M:
            m_composition->nextColorStyle(); 
            break;
        case GLFW_KEY_N:
            m_near_mode = !m_near_mode ; 
            break;
        case GLFW_KEY_O:
            m_scene->nextRenderStyle(modifiers);
            LOG(info) << "Interactor::key_pressed O nextRenderStyle " ; 
            break;
        case GLFW_KEY_P:
            m_scene->nextPhotonStyle(); 
            break;
        case GLFW_KEY_Q:
            m_scene->nextGlobalStyle(); 
            break;
        case GLFW_KEY_R:
            m_rotate_mode = !m_rotate_mode ; 
            break;
        case GLFW_KEY_S:
            m_scale_mode = !m_scale_mode ; 
            break;
        case GLFW_KEY_T:
            m_composition->nextViewMode(modifiers) ; 
            break;
        case GLFW_KEY_U:
            m_composition->nextViewType(modifiers) ; 
            break;
        case GLFW_KEY_V:
            m_composition->nextRotatorMode(modifiers) ; 
            break;
        case GLFW_KEY_W:
            nextOptiXResolutionScale(modifiers); 
            break;
        case _pan_mode_key:
            m_pan_mode = !m_pan_mode ; 
            break;
        case GLFW_KEY_Y:
            m_yfov_mode = !m_yfov_mode ; 
            break;
        case GLFW_KEY_Z:
            m_zoom_mode = !m_zoom_mode ; 
            break;
        case GLFW_KEY_UP:
            m_dragfactor *= 2. ; 
            break;
        case GLFW_KEY_DOWN:
            m_dragfactor *= 0.5 ; 
            break;
        case GLFW_KEY_0:
        case GLFW_KEY_1:
        case GLFW_KEY_2:
        case GLFW_KEY_3:
        case GLFW_KEY_4:
        case GLFW_KEY_5:
        case GLFW_KEY_6:
        case GLFW_KEY_7:
        case GLFW_KEY_8:
        case GLFW_KEY_9:
            number_key_pressed(key - GLFW_KEY_0);
            break; 
        case GLFW_KEY_SPACE:
            space_pressed();
            break;
    } 
    updateStatus();
}




unsigned int Interactor::getModifiers()
{
    unsigned int modifiers = 0 ;
    if( m_keys_down[GLFW_KEY_LEFT_SHIFT]   || m_keys_down[GLFW_KEY_RIGHT_SHIFT] )    modifiers += OpticksConst::e_shift ;
    if( m_keys_down[GLFW_KEY_LEFT_CONTROL] || m_keys_down[GLFW_KEY_RIGHT_CONTROL] )  modifiers += OpticksConst::e_control ;
    if( m_keys_down[GLFW_KEY_LEFT_ALT]     || m_keys_down[GLFW_KEY_RIGHT_ALT] )      modifiers += OpticksConst::e_option ;
    if( m_keys_down[GLFW_KEY_LEFT_SUPER]   || m_keys_down[GLFW_KEY_RIGHT_SUPER] )    modifiers += OpticksConst::e_command ;
    return modifiers ; 
}








/*

 /usr/local/env/graphics/glfw/glfw-3.1.1/include/GLFW/glfw3.h 

                                                     mac keyboard
 382 #define GLFW_KEY_LEFT_SHIFT         340            "shift"
 383 #define GLFW_KEY_LEFT_CONTROL       341            "control"
 384 #define GLFW_KEY_LEFT_ALT           342            "option"
 385 #define GLFW_KEY_LEFT_SUPER         343            "command"
 386 #define GLFW_KEY_RIGHT_SHIFT        344
 387 #define GLFW_KEY_RIGHT_CONTROL      345
 388 #define GLFW_KEY_RIGHT_ALT          346
 389 #define GLFW_KEY_RIGHT_SUPER        347


*/


void Interactor::key_released(unsigned int key)
{
    if(key < NUM_KEYS) m_keys_down[key] = false ; 
    switch (key)
    {
        case GLFW_KEY_0:
        case GLFW_KEY_1:
        case GLFW_KEY_2:
        case GLFW_KEY_3:
        case GLFW_KEY_4:
        case GLFW_KEY_5:
        case GLFW_KEY_6:
        case GLFW_KEY_7:
        case GLFW_KEY_8:
        case GLFW_KEY_9:
            number_key_released(key - GLFW_KEY_0);
            break; 
    } 
}

void Interactor::number_key_pressed(unsigned int number)
{
    m_bookmark_mode = true ; 

    unsigned int modifiers = getModifiers() ;

    m_composition->commitView(); // fold rotator+trackball into view (and home rotator+trackball)

    m_bookmarks->number_key_pressed(number, modifiers);
}

void Interactor::space_pressed()
{
    unsigned int current = m_bookmarks->getCurrent();
    if(current == 0) return ; 
    LOG(info) << "Interactor::space_pressed current " << current ;   

    m_composition->commitView(); // fold rotator+trackball into view (and home rotator+trackball)
    m_bookmarks->updateCurrent();
}


void Interactor::number_key_released(unsigned int number)
{
    m_bookmarks->number_key_released(number);
    m_bookmark_mode = false ; 
}

void Interactor::updateStatus()
{
    snprintf(m_status, STATUS_SIZE , "%s (%u) %s%s%s%s%s%s%s%s %10.3f %u col:%s geo:%s rec:%s ",
           m_bookmarks->getTitle(),
           m_bookmarks->getCurrent(),
           m_zoom_mode ? "z" : "-",
           m_pan_mode  ? "x" : "-",
           m_far_mode  ? "f" : "-",
           m_near_mode ? "n" : "-",
           m_yfov_mode ? "y" : "-",
           m_scale_mode ? "s" : "-",
           m_rotate_mode ? "r" : "-",
           //m_optix_mode ? "o" : "-",
           m_gui_mode ? "g" : "-",
           m_dragfactor,
           m_container,
           m_composition->getColorStyleName(),          
           m_composition->getGeometryStyleName(),          
           m_scene->getRecordStyleName()
           );
}

const char* Interactor::getStatus()
{
    return m_status ;
}

void Interactor::Print(const char* msg)
{
    updateStatus();
    printf("%s %s\n", msg, getStatus() );
}


