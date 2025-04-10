// op --gitemindex



// npy-
#include "PhotonsNPY.hpp"
#include "Index.hpp"
#include "Types.hpp"

// ggeo-
#include "GItemIndex.hh"


#include "Photons.hh"


#ifdef GUI_
#include <imgui.h>
#endif

#include "GUI.hh"


Photons::Photons(Types* types, GItemIndex* boundaries, GItemIndex* seqhis, GItemIndex* seqmat)
    :
    m_types(types),
    m_boundaries(boundaries),
    m_seqhis(seqhis),
    m_seqmat(seqmat)
{
}


void Photons::gui()
{
#ifdef GUI_

    if(m_types)
    {
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Photon Flag Selection"))
        {
            gui_flag_selection();
        }
    }

    if(m_boundaries)
    {
        ImGui::Spacing();
        GUI::gui_radio_select(m_boundaries);
    }

    if(m_seqhis)
    {
        ImGui::Spacing();
        GUI::gui_radio_select(m_seqhis);
    }

    if(m_seqmat)
    {
        ImGui::Spacing();
        GUI::gui_radio_select(m_seqmat);
    }
#endif
}


void Photons::gui_flag_selection()
{
#ifdef GUI_
    // huh not replaced by OpticksAttrSeq ? 
    std::vector<std::string>& labels = m_types->getFlagLabels();  
    bool* flag_selection = m_types->getFlagSelection();
    for(unsigned int i=0 ; i < labels.size() ; i++)
    {
        ImGui::Checkbox(labels[i].c_str(), flag_selection+i );
    }
#endif
}


