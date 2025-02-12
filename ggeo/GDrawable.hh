#pragma once

#include <vector>
#include "GBuffer.hh"
#include "GVector.hh"

template <typename T> class NPY ;

#include "GGEO_API_EXPORT.hh"

class GGEO_API GDrawable {
  public:
      virtual ~GDrawable(){}

      virtual GBuffer* getVerticesBuffer() = 0;
      virtual GBuffer* getNormalsBuffer() = 0;
      virtual GBuffer* getColorsBuffer() = 0;
      virtual GBuffer* getTexcoordsBuffer() = 0;
      virtual GBuffer* getIndicesBuffer() = 0;
      virtual GBuffer* getNodesBuffer() = 0;
      virtual GBuffer* getTransformsBuffer() = 0;
      virtual GBuffer* getIdentityBuffer() = 0;

      virtual NPY<float>*        getITransformsBuffer() = 0;
      virtual NPY<unsigned int>* getInstancedIdentityBuffer() = 0;

      virtual GBuffer* getBoundariesBuffer() = 0;
      virtual GBuffer* getModelToWorldBuffer() = 0;
      virtual std::vector<unsigned int>& getDistinctBoundaries() = 0;

      virtual gfloat4 getCenterExtent(unsigned int index) = 0 ;
      virtual gbbox   getBBox(unsigned int index) = 0 ;
      virtual unsigned int findContainer(gfloat3 p) = 0 ;

};      




