Polyhedron
=============

Relation Between Vertices, Edges and Faces
--------------------------------------------

* http://en.wikipedia.org/wiki/Euler_characteristic  

For a convex polyhedron: `V - E + F = 2`


VRML2FILE Polyhedron Export
----------------------------

external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc::

    169 void G4VRML2SCENEHANDLER::AddPrimitive(const G4Polyhedron& polyhedron)
    170 {
    171 #if defined DEBUG_SCENE_FUNC
    172     G4cerr << "***** AddPrimitive(G4Polyhedron)" << "\n";
    173 #endif
    174 
    175     if (polyhedron.GetNoFacets() == 0) return;
    ...
    199     fDest << "\t"; fDest << "Shape {" << "\n";
    200 
    201     SendMaterialNode();
    202 
    203     fDest << "\t\t" << "geometry IndexedFaceSet {" << "\n";
    204 
    205     fDest << "\t\t\t"   << "coord Coordinate {" << "\n";
    206     fDest << "\t\t\t\t" <<      "point [" << "\n";
    207     G4int i, j;
    208     for (i = 1, j = polyhedron.GetNoVertices(); j; j--, i++) {
    209         G4Point3D point = polyhedron.GetVertex(i);
    210 
    211         point.transform( *fpObjectTransformation );
    212 
    213         fDest << "\t\t\t\t\t";
    214         fDest <<                   point.x() << " ";
    215         fDest <<                   point.y() << " ";
    216         fDest <<                   point.z() << "," << "\n";
    217     }
    218     fDest << "\t\t\t\t" <<      "]" << "\n"; // point
    219     fDest << "\t\t\t"   << "}"      << "\n"; // coord
    220 
    221     fDest << "\t\t\t"   << "coordIndex [" << "\n";
    222 
    223     // facet loop
    224     G4int f;
    225     for (f = polyhedron.GetNoFacets(); f; f--) {
    226 
    227         // edge loop
    228         G4bool notLastEdge;
    229         G4int index = -1, edgeFlag = 1;
    230         fDest << "\t\t\t\t";
    231         do {
    232             notLastEdge = polyhedron.GetNextVertexIndex(index, edgeFlag);
    233             fDest << index - 1 << ", ";
    234         } while (notLastEdge);
    235         fDest << "-1," << "\n";
    236     }
    237     fDest << "\t\t\t"   << "]" << "\n"; // coordIndex
    238 
    239     fDest << "\t\t\t"   << "solid FALSE" << "\n"; // draw backfaces
    240 
    241     fDest << "\t\t" << "}"     << "\n"; // IndexFaceSet
    242     fDest << "\t" << "}"       << "\n"; // Shape




/data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/graphics_reps/src/HepPolyhedron.cc::

    858 bool HepPolyhedron::GetNextVertexIndex(int &index, int &edgeFlag) const
    859 /***********************************************************************
    860  *                                                                     *
    861  * Name: HepPolyhedron::GetNextVertexIndex          Date:    03.09.96  *
    862  * Author: Yasuhide Sawada                          Revised:           *
    863  *                                                                     *
    864  * Function:                                                           *
    865  *                                                                     *
    866  ***********************************************************************/
    867 {
    868   static int iFace = 1;
    869   static int iQVertex = 0;
    870   int vIndex = pF[iFace].edge[iQVertex].v;
    871 
    872   edgeFlag = (vIndex > 0) ? 1 : 0;
    873   index = std::abs(vIndex);
    874 
    875   if (iQVertex >= 3 || pF[iFace].edge[iQVertex+1].v == 0) {
    876     iQVertex = 0;
    877     if (++iFace > nface) iFace = 1;
    878     return false;  // Last Edge
    879   }else{
    880     ++iQVertex;
    881     return true;  // not Last Edge
    882   }
    883 }


* http://en.wikipedia.org/wiki/Polyhedron


`../include/HepPolyhedron.h`::

    168 #ifndef HEP_POLYHEDRON_HH
    169 #define HEP_POLYHEDRON_HH
    170 
    171 #include <CLHEP/Geometry/Point3D.h>
    172 #include <CLHEP/Geometry/Normal3D.h>
    173 
    174 #ifndef DEFAULT_NUMBER_OF_STEPS
    175 #define DEFAULT_NUMBER_OF_STEPS 24
    176 #endif
    177 
    178 class G4Facet {
    179   friend class HepPolyhedron;
    180   friend std::ostream& operator<<(std::ostream&, const G4Facet &facet);
    181 
    182  private:
    183   struct G4Edge { int v,f; };
    184   G4Edge edge[4];          
    ###     a facet can have either 3 or 4 edges
    185 
    186  public:
    187   G4Facet(int v1=0, int f1=0, 
                  int v2=0, int f2=0,
    188           int v3=0, int f3=0, 
                  int v4=0, int f4=0)
    189   { edge[0].v=v1; edge[0].f=f1; 
            edge[1].v=v2; edge[1].f=f2;
    190     edge[2].v=v3; edge[2].f=f3; 
            edge[3].v=v4; edge[3].f=f4; }
    191 };
    192 
    193 class HepPolyhedron {
    194   friend std::ostream& operator<<(std::ostream&, const HepPolyhedron &ph);
    195 
    196  protected:
    197   static int fNumberOfRotationSteps;
    198   int nvert, nface;
    199   HepGeom::Point3D<double> *pV;
    200   G4Facet    *pF;
    201 


::

   1260 double HepPolyhedron::GetVolume() const
   1261 /***********************************************************************
   1262  *                                                                     *
   1263  * Name: HepPolyhedron::GetVolume                   Date:    25.05.01  *
   1264  * Author: E.Chernyaev                              Revised:           *
   1265  *                                                                     *
   1266  * Function: Returns volume of the polyhedron.                         *
   1267  *                                                                     *
   1268  ***********************************************************************/
   1269 {
   1270   double v = 0.;
   1271   for (int iFace=1; iFace<=nface; iFace++) {
   1272     int i0 = std::abs(pF[iFace].edge[0].v);
   1273     int i1 = std::abs(pF[iFace].edge[1].v);
   1274     int i2 = std::abs(pF[iFace].edge[2].v);
   1275     int i3 = std::abs(pF[iFace].edge[3].v);
   1276     Point3D<double> g;
   1277     if (i3 == 0) {
   1278       i3 = i0;
   1279       g  = (pV[i0]+pV[i1]+pV[i2]) * (1./3.);
   1280     }else{
   1281       g  = (pV[i0]+pV[i1]+pV[i2]+pV[i3]) * 0.25;
   1282     }
   1283     v += ((pV[i2] - pV[i0]).cross(pV[i3] - pV[i1])).dot(g);
   1284   }
   1285   return v/6.;
   1286 }



::

   1065 void HepPolyhedron::GetFacet(int iFace, int &n, int *iNodes,
   1066                             int *edgeFlags, int *iFaces) const
   1067 /***********************************************************************
   1068  *                                                                     *
   1069  * Name: HepPolyhedron::GetFacet                    Date:    15.12.99  *
   1070  * Author: E.Chernyaev                              Revised:           *
   1071  *                                                                     *
   1072  * Function: Get face by index                                         *
   1073  *                                                                     *
   1074  ***********************************************************************/
   1075 {
   1076   if (iFace < 1 || iFace > nface) {
   1077     std::cerr
   1078       << "HepPolyhedron::GetFacet: irrelevant index " << iFace
   1079       << std::endl;
   1080     n = 0;
   1081   }else{
   1082     int i, k;
   1083     for (i=0; i<4; i++) {
   1084       k = pF[iFace].edge[i].v;
   1085       if (k == 0) break;
   1086       if (iFaces != 0) iFaces[i] = pF[iFace].edge[i].f;
   1087       if (k > 0) {
   1088         iNodes[i] = k;
   1089         if (edgeFlags != 0) edgeFlags[i] = 1;
   1090       }else{
   1091         iNodes[i] = -k;
   1092         if (edgeFlags != 0) edgeFlags[i] = -1;
   1093       }
   1094     }
   1095     n = i;
   1096   }
   1097 }

::

    270 void HepPolyhedron::CreatePrism()
    271 /***********************************************************************
    272  *                                                                     *
    273  * Name: HepPolyhedron::CreatePrism                  Date:    15.07.96 *
    274  * Author: E.Chernyaev (IHEP/Protvino)               Revised:          *
    275  *                                                                     *
    276  * Function: Set facets for a prism                                    *
    277  *                                                                     *
    278  ***********************************************************************/
    279 {
    280   enum {DUMMY, BOTTOM, LEFT, BACK, RIGHT, FRONT, TOP};
    281 
    282   pF[1] = G4Facet(1,LEFT,  4,BACK,  3,RIGHT,  2,FRONT);
    283   pF[2] = G4Facet(5,TOP,   8,BACK,  4,BOTTOM, 1,FRONT);
    284   pF[3] = G4Facet(8,TOP,   7,RIGHT, 3,BOTTOM, 4,LEFT);
    285   pF[4] = G4Facet(7,TOP,   6,FRONT, 2,BOTTOM, 3,BACK);
    286   pF[5] = G4Facet(6,TOP,   5,LEFT,  1,BOTTOM, 2,RIGHT);
    287   pF[6] = G4Facet(5,FRONT, 6,RIGHT, 7,BACK,   8,LEFT);
    288 }

::

   1388 HepPolyhedronTrd2::HepPolyhedronTrd2(double Dx1, double Dx2,
   1389                                      double Dy1, double Dy2,
   1390                                      double Dz)
   1391 /***********************************************************************
   1392  *                                                                     *
   1393  * Name: HepPolyhedronTrd2                           Date:    22.07.96 *
   1394  * Author: E.Chernyaev (IHEP/Protvino)               Revised:          *
   1395  *                                                                     *
   1396  * Function: Create GEANT4 TRD2-trapezoid                              *
   1397  *                                                                     *
   1398  * Input: Dx1 - half-length along X at -Dz           8----7            *
   1399  *        Dx2 - half-length along X ay +Dz        5----6  !            *
   1400  *        Dy1 - half-length along Y ay -Dz        !  4-!--3            *
   1401  *        Dy2 - half-length along Y ay +Dz        1----2               *
   1402  *        Dz  - half-length along Z                                    *
   1403  *                                                                     *
   1404  ***********************************************************************/
   1405 {
   1406   AllocateMemory(8,6);
   1407 
   1408   pV[1] = Point3D<double>(-Dx1,-Dy1,-Dz);
   1409   pV[2] = Point3D<double>( Dx1,-Dy1,-Dz);
   1410   pV[3] = Point3D<double>( Dx1, Dy1,-Dz);
   1411   pV[4] = Point3D<double>(-Dx1, Dy1,-Dz);
   1412   pV[5] = Point3D<double>(-Dx2,-Dy2, Dz);
   1413   pV[6] = Point3D<double>( Dx2,-Dy2, Dz);
   1414   pV[7] = Point3D<double>( Dx2, Dy2, Dz);
   1415   pV[8] = Point3D<double>(-Dx2, Dy2, Dz);
   1416 
   1417   CreatePrism();
   1418 }




