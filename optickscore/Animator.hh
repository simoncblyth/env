#pragma once

// TODO: try to support live changing of the range 

#define ANIMATOR_DEBUG 1
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API Animator {
    public:
        friend class GUI ; 
    public:
        static const int period_low ; 
        static const int period_high ; 

        static const char* OFF_ ; 
        static const char* SLOW_ ; 
        static const char* NORM_ ; 
        static const char* FAST_ ; 

        typedef enum {  OFF, SLOW, NORM, FAST, NUM_MODE } Mode_t ;


        Animator(float* target, unsigned int period, float low=0.f, float high=1.f);

        void setModeRestrict(Mode_t restrict_);
        bool isModeChanged(Mode_t prior);    
        void modeTransition(float fraction);

        bool isSlowEnabled();
        bool isNormEnabled();
        bool isFastEnabled();

        void home();
        void reset();
        bool step(bool& bump); 
        void Summary(const char* msg);
        void scrub_to(float x, float y, float dx, float dy); // Interactor:K scrub_mode

        float* getTarget(); 
        float getFraction();
        float getFractionFromTarget();
        float getLow(); 
        float getHigh(); 
        bool isActive();

        Mode_t getMode();
        int* getModePtr();
        unsigned int getNumMode();
        void setMode( Mode_t mode);
        void nextMode(unsigned int modifiers);
        const char* getModeString();

        char* description();

    private:
       // used for scrubbing
        void          setTargetValue(float value);
        void          setFraction(float f);
    private:
        void          setTarget(float* target); // qty to be stepped
        unsigned int  getIndex();
        float         getValue();
        float         getFractionForValue(float value);
        unsigned int  find_closest_index(float f);
        bool          isBump();

    private:
        float* make_fractions(unsigned int num);

    private:
        Mode_t       m_mode ; 
        Mode_t       m_restrict ; 
        unsigned int m_period[NUM_MODE] ; 
        float        m_low ; 
        float        m_high ; 
        float*       m_fractions[NUM_MODE] ; 
        unsigned int m_count ; 
        unsigned int m_index ; 
        char         m_desc[32] ; 
        float*       m_target ; 
        int          m_increment ; 
};



