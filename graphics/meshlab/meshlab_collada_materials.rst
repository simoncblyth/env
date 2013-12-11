MeshLab Collada Materials
===========================

Appears to be recognize textures based effects only, not phong parameterized effects.

/usr/local/env/graphics/meshlab/vcglib/wrap/dae/util_dae.h::


    090     class UtilDAE
    091     {
    ...
    569         struct WedgeAttribute
    570         {
    571             QDomNode wnsrc;    //  input/sematic/NORMAL
    572             QStringList wn;
    573             int offnm;
    574 
    575             QDomNode wtsrc;    // input/semantic/TEXCOORD
    576             QStringList wt;
    577             int stride;
    578             int offtx;
    579 
    580             QDomNode wcsrc;    // input/semantic/COLOR
    581             QStringList wc;
    582             int offcl;
    583         };
    584     };


::

    g4pb:wrap blyth$ find . -name '*.h' -exec grep -H WedgeAttribute {} \;
    ./dae/util_dae.h:                       void usePerWedgeAttributes(PERWEDGEATTRIBUTETYPE att,const unsigned int multitexture = 1,const unsigned int multicolor = 1)
    ./dae/util_dae.h:               struct WedgeAttribute
    ./io_trimesh/import_dae.h:              static void FindStandardWedgeAttributes(WedgeAttribute& wed,const QDomNode nd,const QDomDocument doc)
    ./io_trimesh/import_dae.h:                                      WedgeAttribute wa;
    ./io_trimesh/import_dae.h:                    FindStandardWedgeAttributes(wa,polylist.at(tript),*(info.doc));
    ./io_trimesh/import_dae.h:                              WedgeAttribute wa;
    ./io_trimesh/import_dae.h:                FindStandardWedgeAttributes(wa,polylist.at(pl),*(info.doc));
    ./io_trimesh/import_dae.h:                                      WedgeAttribute wa;
    ./io_trimesh/import_dae.h:                    FindStandardWedgeAttributes(wa,triNodeList.at(tript),*(info.doc));
    g4pb:wrap blyth$ 


::

     142         static void FindStandardWedgeAttributes(WedgeAttribute& wed,const QDomNode nd,const QDomDocument doc)
     143         {
     144             wed.wnsrc = findNodeBySpecificAttributeValue(nd,"input","semantic","NORMAL");
     145             wed.offnm = findStringListAttribute(wed.wn,wed.wnsrc,nd,doc,"NORMAL");
     146 
     147             wed.wtsrc = findNodeBySpecificAttributeValue(nd,"input","semantic","TEXCOORD");
     148             if (!wed.wtsrc.isNull())
     149             {
     150                 QDomNode src = attributeSourcePerSimplex(nd,doc,"TEXCOORD");
     151                 if (isThereTag(src,"accessor"))
     152                 {
     153                     QDomNodeList wedatts = src.toElement().elementsByTagName("accessor");
     154                     wed.stride = wedatts.at(0).toElement().attribute("stride").toInt();
     155                 }
     156                 else
     157                     wed.stride = 2;
     158             }
     159             else
     160                 wed.stride = 2;
     161 
     162             wed.offtx = findStringListAttribute(wed.wt,wed.wtsrc,nd,doc,"TEXCOORD");  // fills wed.wt 
     163 
     164             wed.wcsrc = findNodeBySpecificAttributeValue(nd,"input","semantic","COLOR");
     165             wed.offcl = findStringListAttribute(wed.wc,wed.wcsrc,nd,doc,"COLOR");     // fill wed.wc 
     166         }



::

    482         inline static int findStringListAttribute(QStringList& list,const QDomNode node,const QDomNode poly,const QDomDocument startpoint,const char* token)
    483         {
    484             int offset = 0;
    485             if (!node.isNull())
    486             {
    487                 offset = node.toElement().attribute("offset").toInt();
    488                 QDomNode st = attributeSourcePerSimplex(poly,startpoint,token);  // lookup input elem with matched token (semantic) use the source reference to access the data 
    489                 valueStringList(list,st,"float_array");
    490             }
    491             return offset;
    492         }
    493 
    494 


::

     173         static DAEError LoadPolygonalListMesh(QDomNodeList& polylist,ColladaMesh& m,const size_t offset,InfoDAE& info,QMap<QString,QString> &materialBinding)
     174         {
     175             if(polylist.isEmpty()) return E_NOERROR;
     176             QDEBUG("****** LoadPolygonalListMesh (initial mesh size %i %i)",m.vert.size(),m.fn);
     177             for(int tript = 0; tript < polylist.size();++tript)
     178             {
     ...
     218                     WedgeAttribute wa;
     219                     FindStandardWedgeAttributes(wa,polylist.at(tript),*(info.doc));
     220                     QDEBUG("*******                 Start Reading faces. Attributes Offsets: offtx %i - offnm %i - offcl %i",wa.offtx,wa.offnm,wa.offcl);
     221 
     222           int faceIndexCnt=0;
     223                     int jj = 0;
     224                     for(int ff = 0; ff < (int) faceSizeList.size();++ff) // for each polygon
     ///  from "vcount" eg 3 3 3 3 3 4 4 4 4 4
     225                     {
     226                         int curFaceVertNum = faceSizeList.at(ff).toInt();
     227 
     228                         MyPolygon<typename ColladaMesh::VertexType>  polyTemp(curFaceVertNum);
     229                         for(int tt = 0;tt < curFaceVertNum ;++tt)  // for each vertex of the polygon
     230                         {
     231                             int indvt = faceIndexList.at(faceIndexCnt).toInt();
     ///  from "p" 
     232                             if(faceSizeList.size()<100) QDEBUG("*******                 Reading face[%3i].V(%i) = %4i  (%i-th of the index list) (face has %i vertices)",ff,tt,indvt,faceIndexCnt,curFaceVertNum);
     233                             assert(indvt + offset < m.vert.size());
     234                             polyTemp._pv[tt] = &(m.vert[indvt + offset]);
     235                             faceIndexCnt +=faceAttributeNum;
     236 
     237                             WedgeTextureAttribute(polyTemp._txc[tt],faceIndexList,ind_txt, wa.wt ,wa.wtsrc, jj + wa.offtx,wa.stride);
     ///
     ///      associates a vertex, with texture index  
     ///               jj + wa.offtx ==>  faceind     points to indtx from faceIndexList  
     ///
     ///      giving association with texture 
     ///              U(), V()    coordinates within the image
     ///              N()         image number  
     ///                 
     ///
     249 
     250                             jj += faceAttributeNum;
     251                         }
     252 
     253                         AddPolygonToMesh(polyTemp,m);
     254                     }
     255                 }





::

     110         // this one is used for the polylist nodes
     111         static int WedgeTextureAttribute(typename ColladaMesh::FaceType::TexCoordType & WT, const QStringList faceIndexList, int ind_txt, const QStringList wt, const QDomNode wtsrc,const int faceind,const int stride = 2)
     112         {
     113             int indtx = -1;
     114             if (!wtsrc.isNull())
     115             {
     116                 indtx = faceIndexList.at(faceind).toInt();
     117                 //int num = wt.size(); 
     118                 assert(indtx * stride < wt.size());
     119                 WT = vcg::TexCoord2<float>();
     120                 WT.U() = wt.at(indtx * stride).toFloat();
     121                 WT.V() = wt.at(indtx * stride + 1).toFloat();
     122                 WT.N() = ind_txt;
     123             }
     124             return indtx;
     125         }



Untextured Polylist::

    127         <polylist count="30" material="WHITE">
    128           <input offset="0" semantic="VERTEX" source="#near_hall_top_dwarf0xbbd1a68-Vtx" />
    129           <input offset="1" semantic="NORMAL" source="#near_hall_top_dwarf0xbbd1a68-Norm" />
    130           <vcount>4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 </vcount>
    131           <p>0 0  
                     3 0  
                     2 0  
                     1 0   
                     6 1  
                     5 1  
                     1 1  
                     2 1   
                     5 2  4 2  0 2  1 2   4 3  5 3  6 3  7 3   11 4  13 4  10 4  8 4   13 5  12 5  9 5  10 5   15 6  14 6  3 6   4 7  7 7  17 7   15 8  3 8  0 8   4 9  17 9  16 9   16 10  15 10  0 10   0     11  4 11  16 11   19 12  17 12  7 12   7 13  6 13  2 13   3 14  14 14  18 14   19 15  7 15  2 15   2 16  3 16  18 16   18 17  19 17  2 17   14 18  15 18  8 18   9 19  18 19  14 19   14 20  8 20  10 20   10 21  9 21  14 21   19 22  18 22      9 22   9 23  12 23  19 23   15 24  16 24  11 24   11 25  8 25  15 25   17 26  19 26  12 26   13 27  11 27  16 27   17 28  12 28  13 28   13 29  16 29  17 29   
                   </p>
    132         </polylist>



::

     <input semantic="TEXCOORD" source="#rectangle_object-mesh-map-channel1" offset="2" set="1"/>



One texture image per material (only ~30 materials) leads to meshlab loading 4522 texture images 
and being unusably slow.::

    LOG: 0 Loading textures
    LOG: 0  Texture[ 4515 ] =  './textures/OpaqueVacuum.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )
    LOG: 0 Loading textures
    LOG: 0  Texture[ 4516 ] =  './textures/Pyrex.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )
    LOG: 0 Loading textures
    LOG: 0  Texture[ 4517 ] =  './textures/Bialkali.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )
    LOG: 0 Loading textures
    LOG: 0  Texture[ 4518 ] =  './textures/OpaqueVacuum.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )
    LOG: 0 Loading textures
    LOG: 0  Texture[ 4519 ] =  './textures/Pyrex.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )
    LOG: 0 Loading textures
    LOG: 0  Texture[ 4520 ] =  './textures/Bialkali.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )
    LOG: 0 Loading textures
    LOG: 0  Texture[ 4521 ] =  './textures/OpaqueVacuum.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )
    LOG: 0 Loading textures
    LOG: 0  Texture[ 4522 ] =  './textures/RadRock.png' (    100 x    100 ) -> (    128 x    128 )
            will be loaded as GL texture id 0  ( 128 x 128 )



