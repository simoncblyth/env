#pragma once

#include <glm/fwd.hpp>
#include <vector>
#include <string>

// g4-
class G4VPhysicalVolume ;
class G4LogicalVolume ;
class G4Material ; 
class G4VSolid;

#include "G4Transform3D.hh"
#include "G4MaterialPropertyVector.hh" 
// fwd-decl difficult due to typedef 

// okc-
class OpticksQuery ; 

// npy-
template <typename T> class NPY ;
class NBoundingBox ;


// TODO: get rid of the VolumeTreeTraverse


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CTraverser {
    public:
        static const char* GROUPVEL ; 
    public:
        // need-to-know-basis: leads to more focussed, quicker to understand and easier to test code
        CTraverser(G4VPhysicalVolume* top, NBoundingBox* bbox, OpticksQuery* query);
    private:
        void init();
    public:
        void Traverse();
        void createGroupVel();
        void setVerbosity(unsigned int verbosity);
        void Summary(const char* msg="CTraverser::Summary"); 
        std::string description();
    private:
         void AncestorTraverse();
         void VolumeTreeTraverse();
    public:
        void         dumpMaterials(const char* msg="CTraverser::dumpMaterials");
        unsigned int getNumMaterials();
        unsigned int getNumMaterialsWithoutMPT();
    public:
        const G4Material* getMaterial(unsigned int index);
        G4Material*       getMaterialWithoutMPT(unsigned int index);
    public:
        const char*  getPVName(unsigned int index);
        glm::mat4    getGlobalTransform(unsigned int index);
        glm::mat4    getLocalTransform(unsigned int index);
        glm::vec4    getCenterExtent(unsigned int index);
    public:
        NPY<float>*  getGlobalTransforms();
        NPY<float>*  getLocalTransforms();
        NPY<float>*  getCenterExtent();
    public:
        unsigned int getNumGlobalTransforms();
        unsigned int getNumLocalTransforms();
        unsigned int getNumSelected();
    private:
        void collectTransformT(NPY<float>* buffer, const G4Transform3D& T);
        void collectTransform(NPY<float>* buffer, const G4Transform3D& T);
    private:
        void AncestorVisit(std::vector<const G4VPhysicalVolume*> ancestors, bool selected);
        void AncestorTraverse(std::vector<const G4VPhysicalVolume*> ancestors, const G4VPhysicalVolume* pv, unsigned int depth, bool recursive_select);
    private:
        void updateBoundingBox(const G4VSolid* solid, const G4Transform3D& transform, bool selected);
    private:
        G4Transform3D VolumeTreeTraverse(const G4LogicalVolume* const volumePtr, const G4int depth);
        void Visit(const G4LogicalVolume* const lv);
        void VisitPV(const G4VPhysicalVolume* const pv, const G4Transform3D& T );

        bool hasMaterial(const G4Material* material) ; 
        void addMaterial(const G4Material* material) ; 
        void dumpMaterial(const G4Material* material);
        void dumpMaterialProperty(const G4String& name, const G4MaterialPropertyVector* pvec);
    private:
        bool hasMaterialWithoutMPT(G4Material* material) ; 
        void addMaterialWithoutMPT(G4Material* material) ; 
    private:
        G4VPhysicalVolume*             m_top ; 
        NBoundingBox*                  m_bbox ; 
        OpticksQuery*                  m_query ; 
        unsigned int                   m_ancestor_index ; 

        std::vector<const G4Material*> m_materials ;
        std::vector<G4Material*>       m_materials_without_mpt ;

        unsigned int   m_verbosity ; 
        unsigned int   m_gcount ; 
        unsigned int   m_lcount ; 

        NPY<float>*    m_gtransforms ; 
        NPY<float>*    m_ltransforms ; 
        NPY<float>*    m_center_extent ;
 
        std::vector<std::string> m_pvnames ; 
        std::vector<unsigned int> m_selection ; 
};

#include "CFG4_TAIL.hh"


