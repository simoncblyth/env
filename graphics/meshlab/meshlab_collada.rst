Meshlab Collada Importer Abysmal Speed
==========================================

.. contents:: :local:


Collader Import takes 41 min for full geometry on G 
-------------------------------------------------------

Pycollada using numpy takes maybe 40 s.  C++ Qt meshlab taking 41 min. 

::

    In [50]: 2494335./1000./60.
    Out[50]: 41.572250000000004


::

    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '0.707107 -0.707107 0 6603.82 0.707107 0.707107 0 3603.82 0 0 1 0 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '6.12303e-17 -1 0 0 1 6.12303e-17 0 5150 0 0 1 0 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '-0.707107 -0.707107 0 -6603.82 0.707107 -0.707107 0 3603.82 0 0 1 0 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '-1 -1.22461e-16 0 -8150 1.22461e-16 -1 0 0 0 0 1 0 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '-0.707107 0.707107 0 -6603.82 -0.707107 -0.707107 0 -3603.82 0 0 1 0 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '6.12303e-17 1 0 0 -1 6.12303e-17 0 -5150 0 0 1 0 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '0.707107 0.707107 0 6603.82 -0.707107 0.707107 0 -3603.82 0 0 1 0 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    Parsing matrix node; text value is '1 0 0 0 0 1 0 0 0 0 1 -5150 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    ====== searching among library_effects the effect with id '__dd__Materials__RadRock_fx_0xca9e180' 
    LOG: 0 Opened mesh /usr/local/env/geant4/geometry/gdml/20131119-1632/g4_00.dae in 2494335 msec
    LOG: 0 All files opened in 2518742 msec


11 min after Geometry Cache addition, but exploded! 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Hmm, its faster but geometry is "exploded" into little pieces.

::

    ******                 Reading face[ 95].V(0) =   45  (570-th of the index list) (face has 3 vertices)
    *******                 Reading face[ 95].V(1) =   43  (572-th of the index list) (face has 3 vertices)
    *******                 Reading face[ 95].V(2) =   40  (574-th of the index list) (face has 3 vertices)
    ****** LoadPolygonalListMesh (final  mesh size vn 50 vertsize 50 - fn 192 facesize 192)
    **** Loading a Geometry Mesh **** (final   mesh size 50 50 - 192 192)
    ** instance_geometry with url #near-radslab-box-90xc8e73c0 (final mesh size 1246046 1246046 - 2452916 2452916)
    LOG: 0 Opened mesh /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae in 657315 msec
    LOG: 0 All files opened in 663948 msec


::

    In [7]: 2518742./663948.
    Out[7]: 3.7935832324218164

    In [8]: 663948./1000./60.
    Out[8]: 11.065799999999999



Why is collada importer so slow ?
------------------------------------

/usr/local/env/graphics/meshlab/meshlab/src/meshlabplugins/io_collada/io_collada.cpp::

    104 bool ColladaIOPlugin::open(const QString &formatName, const QString &fileName, MeshModel &m, int& mask, const RichParameterSet &, CallBackPos *cb, QWidget *parent)
    ...
    118     if(formatName.toUpper() == tr("DAE"))
    119     {
    ...
    121         tri::io::InfoDAE  info;
    122         if (!tri::io::ImporterDAE<CMeshO>::LoadMask(filename.c_str(), info))
    123             return false;
    ...
    129         int result = vcg::tri::io::ImporterDAE<CMeshO>::Open(m.cm, filename.c_str(),info);


/usr/local/env/graphics/meshlab/vcglib/wrap/io_trimesh/import_dae.h::

      25 #define __VCGLIB_IMPORTERDAE
      26 
      27 //importer for collada's files
      28 
      29 #include <wrap/dae/util_dae.h>
      30 
      31 // uncomment one of the following line to enable the Verbose debugging for the parsing
      32 #define QDEBUG if(1) ; else {assert(0);} 
      33 //#define QDEBUG qDebug
      34 
      35 namespace vcg {
      36 namespace tri {
      37 namespace io {
      38     template<typename OpenMeshType>
      39     class ImporterDAE : public UtilDAE
      40     {
      41   public:

/usr/local/env/graphics/meshlab/vcglib/wrap/io_trimesh/import_dae.h::

     713         //merge all meshes in the collada's file in the templeted mesh m
     714         //I assume the mesh 
     715 
     716         static int Open(OpenMeshType& m,const char* filename, InfoDAE& info, CallBackPos *cb=0)
     717         {
     718             (void)cb;
     719 
     720             QDEBUG("----- Starting the processing of %s ------",filename);
     721             //AdditionalInfoDAE& inf = new AdditionalInfoDAE();
     722             //info = new InfoDAE();
     723 
     724             QDomDocument* doc = new QDomDocument(filename);
     725             info.doc = doc;


Code looks like it is not doing any caching, repeatedly searching DOM for for every refernence.

/usr/local/env/graphics/meshlab/vcglib/wrap/dae::


    478         /* Very important procedure 
    479             it has the task to finde the name of the image node corresponding to a given material id, 
    480             it assuemes that the material name that is passed have already been bound with the current bindings  
    481         */
    482 
    483         inline static QDomNode textureFinder(QString& boundMaterialName, QString &textureFileName, const QDomDocument doc)
    484         {
    485             boundMaterialName.remove('#');
    486             //library_material -> material -> instance_effect
    487             QDomNodeList lib_mat = doc.elementsByTagName("library_materials");
    488             if (lib_mat.size() != 1)
    489                 return QDomNode();
    490             QDomNode material = findNodeBySpecificAttributeValue(lib_mat.at(0),QString("material"),QString("id"),boundMaterialName);
    491             if (material.isNull())
    492                 return QDomNode();
    493             QDomNodeList in_eff = material.toElement().elementsByTagName("instance_effect");
    494             if (in_eff.size() == 0)
    495                 return QDomNode();
    496             QString url = in_eff.at(0).toElement().attribute("url");
    497             if ((url.isNull()) || (url == ""))
    498                 return QDomNode();
    499             url = url.remove('#');
    500       qDebug("====== searching among library_effects the effect with id '%s' ",qPrintable(url));
    501             //library_effects -> effect -> instance_effect
    502             QDomNodeList lib_eff = doc.elementsByTagName("library_effects");
    503             if (lib_eff.size() != 1)
    504                 return QDomNode();
    505             QDomNode effect = findNodeBySpecificAttributeValue(lib_eff.at(0),QString("effect"),QString("id"),url);
    506             if (effect.isNull())
    507                 return QDomNode();
    508             QDomNodeList init_from = effect.toElement().elementsByTagName("init_from");
    509             if (init_from.size() == 0)
    510                 return QDomNode();
    511             QString img_id = init_from.at(0).toElement().text();
    512             if ((img_id.isNull()) || (img_id == ""))
    513                 return QDomNode();
    514 
    515             //library_images -> image
    516             QDomNodeList libraryImageNodeList = doc.elementsByTagName("library_images");
    517             qDebug("====== searching among library_images the effect with id '%s' ",qPrintable(img_id));
    518             if (libraryImageNodeList.size() != 1)
    519                 return QDomNode();
    520             QDomNode imageNode = findNodeBySpecificAttributeValue(libraryImageNodeList.at(0),QString("image"),QString("id"),img_id);
    521             QDomNodeList initfromNode = imageNode.toElement().elementsByTagName("init_from");
    522             textureFileName= initfromNode.at(0).firstChild().nodeValue();
    523             qDebug("====== the image '%s' has a %i init_from nodes text '%s'",qPrintable(img_id),initfromNode.size(),qPrintable(textureFileName));
    524 
    525             return imageNode;
    526         }


/usr/local/env/graphics/meshlab/vcglib/wrap/dae/util_dae.h::

    249         inline static QDomNode findNodeBySpecificAttributeValue(const QDomNodeList& ndl,const QString& attrname,const QString& attrvalue)
    250         {
    251             int ndl_size = ndl.size();
    252             int ind = 0;
    253             while(ind < ndl_size)
    254             {
    255                 QString st = ndl.at(ind).toElement().attribute(attrname);
    256                 if (st == attrvalue)
    257                     return ndl.at(ind);
    258                 ++ind;
    259             }
    260             return QDomNode();
    261         }
    262 
    263         inline static QDomNode findNodeBySpecificAttributeValue(const QDomNode n,const QString& tag,const QString& attrname,const QString& attrvalue)
    264         {
    265             return findNodeBySpecificAttributeValue(n.toElement().elementsByTagName(tag),attrname,attrvalue);
    266         }
    267 
    268         inline static QDomNode findNodeBySpecificAttributeValue(const QDomDocument n,const QString& tag,const QString& attrname,const QString& attrvalue)
    269         {
    270             return findNodeBySpecificAttributeValue(n.elementsByTagName(tag),attrname,attrvalue);
    271         }



Separate plugin build output
-----------------------------

First section all in the ``Open(``  /usr/local/env/graphics/meshlab/vcglib/wrap/io_trimesh/import_dae.h::

    ----- Starting the processing of /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae ------
    File Contains 1 Scenes
    Scene 0 contains 1 instance_visual_scene 
    instance_visual_scene 0 refers DefaultScene 
    instance_visual_scene DefaultScene has 1 children
    Processing Visual Scene child 0 - of type 'node'          

Subsequentluy goes recursive /usr/local/env/graphics/meshlab/vcglib/wrap/io_trimesh/import_dae.h ::

     806                         if(node.toElement().tagName()=="node")
     807                         {
     808                             ColladaMesh newMesh;
     809                             AddNodeToMesh(node.toElement(), newMesh, baseTr,info);
     810                             tri::Append<OpenMeshType,ColladaMesh>::Mesh(m,newMesh);
     811                         }


::

     590         static void AddNodeToMesh(QDomElement node,
     591                                                             ColladaMesh& m, Matrix44f curTr,
     592                                                             InfoDAE& info)
     593         {
     594                 QDEBUG("Starting processing <node> with id %s",qPrintable(node.attribute("id")));
     595 
     596                 curTr = curTr * getTransfMatrixFromNode(node);
     597 
     598                 QDomNodeList geomNodeList = node.elementsByTagName("instance_geometry");
     599                 for(int ch = 0;ch < geomNodeList.size();++ch)
     600                 {
     601                     QDomElement instGeomNode= geomNodeList.at(ch).toElement();
     602                     if(instGeomNode.parentNode()==node) // process only direct child
     603                     {
     604                         QDEBUG("** instance_geometry with url %s (intial mesh size %i %i T = %i)",qPrintable(instGeomNode.attribute("url")),m.vn,m.fn,m.textures.size());
     605                         //assert(m.textures.size()>0 == HasPerWedgeTexCoord(m));
     606                         QString geomNode_url;
     607                         referenceToANodeAttribute(instGeomNode,"url",geomNode_url);
     608                         QDomNode refNode = findNodeBySpecificAttributeValue(*(info.doc),"geometry","id",geomNode_url);
     609                         QDomNodeList bindingNodes = instGeomNode.toElement().elementsByTagName("bind_material");
     610                         QMap<QString,QString> materialBindingMap;
     611                         if( bindingNodes.size()>0) {
     612                             QDEBUG("**    instance_geometry has a material binding");
     613                             GenerateMaterialBinding(instGeomNode,materialBindingMap);
     614                         }
     615 

     ////  this is the place to cache the mesh keyed by refNode , is ColladaMesh up to living in a container ?

     616                         ColladaMesh newMesh;
     617 //                      newMesh.face.EnableWedgeTex();
     618                         LoadGeometry(newMesh, info, refNode.toElement(),materialBindingMap);
     619                         tri::UpdatePosition<ColladaMesh>::Matrix(newMesh,curTr);
     620                         tri::Append<ColladaMesh,ColladaMesh>::Mesh(m,newMesh);
     621                         QDEBUG("** instance_geometry with url %s (final mesh size %i %i - %i %i)",qPrintable(instGeomNode.attribute("url")),m.vn,m.vert.size(),m.fn,m.face.size());
     622                     }
     623                 }


/usr/local/env/graphics/meshlab/vcglib/vcg/complex/complex.h::

    415 private:
    416     // TriMesh cannot be copied. Use Append (see vcg/complex/append.h)
    417   TriMesh operator =(const TriMesh &  /*m*/){assert(0);return TriMesh();}
    418     TriMesh(const TriMesh & ){}
    419 



::

    Starting processing <node> with id top
    getTrans form node with tag node
    Found a instance_node with url 'World0xc7ba680'
    Starting processing <node> with id World0xc7ba680
    getTrans form node with tag node
    ** instance_geometry with url #WorldBox0xc8e27e0 (intial mesh size 0 0 T = 0)
    **    instance_geometry has a material binding
    ++++ Found 1 instance_material binding
    ++++++ WHITE -> #__dd__Materials__Vacuum0xbd75258
    **** Loading a Geometry Mesh **** (initial mesh size 0 0)
    ****** LoadPolygonalListMesh (initial mesh size 8 0)
    ******    material id 'WHITE' -> '#__dd__Materials__Vacuum0xbd75258'
    ====== searching among library_effects the effect with id '__dd__Materials__Vacuum_fx_0xbd75258' 
    ******   but we were not able to find the corresponding image node
    *******                 Start Reading faces. Attributes Offsets: offtx 0 - offnm 1 - offcl 0
    *******                 Reading face[  0].V(0) =    0  (0-th of the index list) (face has 4 vertices)
    *******                 Reading face[  0].V(1) =    3  (2-th of the index list) (face has 4 vertices)
    *******                 Reading face[  0].V(2) =    2  (4-th of the index list) (face has 4 vertices)
    *******                 Reading face[  0].V(3) =    1  (6-th of the index list) (face has 4 vertices)
    *******                 Reading face[  1].V(0) =    4  (8-th of the index list) (face has 4 vertices)
    *******                 Reading face[  1].V(1) =    7  (10-th of the index list) (face has 4 vertices)






Before profiling/optimising need to check the SVN future of meshlab/vcglib
----------------------------------------------------------------------------

Sourceforge yuck.

* http://sourceforge.net/p/meshlab/code/6239/log/?path=/trunk

Slow code is actually in vcglib

* http://vcg.isti.cnr.it/~cignoni/newvcglib/html/
* http://sourceforge.net/projects/vcg/
* http://svn.code.sf.net/p/vcg/code/trunk/vcglib/

::

    simon:dae blyth$ vcglib-cd
    simon:vcglib_trunk blyth$ pwd
    /usr/local/env/graphics/vcglib_trunk
    simon:vcglib_trunk blyth$ cd wrap/dae
    simon:dae blyth$ 
    simon:dae blyth$ svn log . -v
    ------------------------------------------------------------------------
    r4985 | granzuglia | 2013-10-25 04:51:03 +0800 (Fri, 25 Oct 2013) | 1 line
    Changed paths:
       M /trunk/vcglib/wrap/dae/poly_triangulator.h

    - added missing include file
    ------------------------------------------------------------------------
    r4983 | granzuglia | 2013-10-25 00:18:13 +0800 (Fri, 25 Oct 2013) | 1 line
    Changed paths:
       M /trunk/vcglib/wrap/dae/colladaformat.h
       A /trunk/vcglib/wrap/dae/poly_triangulator.h
       M /trunk/vcglib/wrap/dae/util_dae.h

    - updated collada format in order to manage alpha channel colour
    ------------------------------------------------------------------------
    r4861 | granzuglia | 2013-03-25 03:51:43 +0800 (Mon, 25 Mar 2013) | 2 lines
    Changed paths:
       M /trunk/vcglib/wrap/dae/util_dae.h
       M /trunk/vcglib/wrap/dae/xmldocumentmanaging.h

    - small changes for qt5.0 compatibility

    ------------------------------------------------------------------------
    r4752 | cignoni | 2012-11-28 06:31:48 +0800 (Wed, 28 Nov 2012) | 1 line
    Changed paths:
       M /trunk/vcglib/wrap/dae/colladaformat.h
       M /trunk/vcglib/wrap/io_trimesh/export_idtf.h

    Added a few missing const specifiers
    ------------------------------------------------------------------------
    r4180 | cignoni | 2011-10-05 23:04:40 +0800 (Wed, 05 Oct 2011) | 1 line
    Changed paths:
       A /trunk/vcglib (from /trunk/vcglib:4178)
       R /trunk/vcglib/apps (from /trunk/vcglib/apps:4178)
       R /trunk/vcglib/apps/metro (from /trunk/vcglib/apps/metro:4178)
       R /trunk/vcglib/apps/metro/defs.h (from /trunk/vcglib/apps/metro/defs.h:4178)
       R /trunk/vcglib/apps/metro/history.txt (from /trunk/vcglib/apps/metro/history.txt:4178)


::

    simon:meshlab blyth$ diff -r --brief $(meshlab-dir)/../../vcglib/wrap/dae $(vcglib-dir)/wrap/dae
    Files /usr/local/env/graphics/meshlab/meshlab/src/../../vcglib/wrap/dae/colladaformat.h and /usr/local/env/graphics/vcglib_trunk/wrap/dae/colladaformat.h differ
    Only in /usr/local/env/graphics/vcglib_trunk/wrap/dae: poly_triangulator.h
    Files /usr/local/env/graphics/meshlab/meshlab/src/../../vcglib/wrap/dae/util_dae.h and /usr/local/env/graphics/vcglib_trunk/wrap/dae/util_dae.h differ
    Files /usr/local/env/graphics/meshlab/meshlab/src/../../vcglib/wrap/dae/xmldocumentmanaging.h and /usr/local/env/graphics/vcglib_trunk/wrap/dae/xmldocumentmanaging.h differ
    simon:meshlab blyth$ 



Fix by addition of a geometry cache
----------------------------------------

Using a ``QCache`` would have been nice for eviction control to restrict memory usage, 
but the ColladaMesh type structure is too contorted to be usable 
outside of its definition class, hence the ``void*`` cheating.

/usr/local/env/graphics/meshlab/vcglib/wrap/dae/util_dae.h::

     52 #if defined SCB_COLLADA_GEOMETRY_CACHE
     53 #include <QHash>
     54 #endif
     55 
     56 namespace vcg {
     57 namespace tri {
     58 namespace io {
     59 
     60 
     61     
     62     class InfoDAE  : public AdditionalInfo
     63     {
     64         public:
     65 
     66         InfoDAE() :AdditionalInfo(){
     67             doc = NULL;
     68             textureIdMap.clear();
     69         }   
     70 
     71         ~InfoDAE(){
     72             if(doc!=NULL) delete doc;
     73 #if defined SCB_COLLADA_GEOMETRY_CACHE
     74             geometry_cache.clear() ; 
     75 #endif  
     76         }
     77 
     78         QDomDocument* doc;
     79         QMap<QString,int> textureIdMap;
     80 
     81 #if defined SCB_COLLADA_GEOMETRY_CACHE
     82        QHash<QString, void* > geometry_cache ;
     83 #endif 
     84 
     85 
     86     };


/usr/local/env/graphics/meshlab/vcglib/wrap/io_trimesh/import_dae.h::

     591         static void AddNodeToMesh(QDomElement node,
     592                                                             ColladaMesh& m, Matrix44f curTr,
     593                                                             InfoDAE& info)
     594         {
     595                 QDEBUG("Starting processing <node> with id %s",qPrintable(node.attribute("id")));
     596 
     597                 curTr = curTr * getTransfMatrixFromNode(node);
     598 
     599                 QDomNodeList geomNodeList = node.elementsByTagName("instance_geometry");
     600                 for(int ch = 0;ch < geomNodeList.size();++ch)
     601                 {
     602                     QDomElement instGeomNode= geomNodeList.at(ch).toElement();
     603                     if(instGeomNode.parentNode()==node) // process only direct child
     604                     {
     605                         QDEBUG("** instance_geometry with url %s (intial mesh size %i %i T = %i)",qPrintable(instGeomNode.attribute("url")),m.vn,m.fn,m.textures.size());
     606                         //assert(m.textures.size()>0 == HasPerWedgeTexCoord(m));
     607                         QString geomNode_url;
     608                         referenceToANodeAttribute(instGeomNode,"url",geomNode_url);
     609 
     610 #if defined SCB_COLLADA_GEOMETRY_CACHE
     611                        // cache them for reuse
     612                         bool geometry_cached = info.geometry_cache.contains(geomNode_url) ;
     613                         ColladaMesh* newMeshPtr ;
     614 
     615                         if( geometry_cached )
     616                         {
     617                             QDEBUG("**   instance_geometry cache hit %s ", qPrintable(geomNode_url));
     618                             newMeshPtr = (ColladaMesh*)info.geometry_cache.value(geomNode_url) ;
     619                         }
     620                         else
     621                         {
     622                             QDEBUG("**   instance_geometry CACHE MISS %s ", qPrintable(geomNode_url));
     623                             newMeshPtr = new ColladaMesh ;
     624 
     625                             QDomNode refNode = findNodeBySpecificAttributeValue(*(info.doc),"geometry","id",geomNode_url);
     626                             QDomNodeList bindingNodes = instGeomNode.toElement().elementsByTagName("bind_material");
     627                             QMap<QString,QString> materialBindingMap;
     628                             if( bindingNodes.size()>0) {
     629                                 QDEBUG("**    instance_geometry has a material binding");
     630                                 GenerateMaterialBinding(instGeomNode,materialBindingMap);
     631                              }
     632 
     633                              LoadGeometry(*newMeshPtr, info, refNode.toElement(),materialBindingMap);
     634                              info.geometry_cache.insert(geomNode_url, (void*)newMeshPtr) ;
     635                         }
     636                         ColladaMesh& newMesh = *newMeshPtr ;
     637 #else
     638 
     639                         QDomNode refNode = findNodeBySpecificAttributeValue(*(info.doc),"geometry","id",geomNode_url);
     640                         QDomNodeList bindingNodes = instGeomNode.toElement().elementsByTagName("bind_material");
     641                         QMap<QString,QString> materialBindingMap;
     642                         if( bindingNodes.size()>0) {
     643                             QDEBUG("**    instance_geometry has a material binding");
     644                             GenerateMaterialBinding(instGeomNode,materialBindingMap);
     645                         }
     646                         ColladaMesh newMesh;
     647 //                      newMesh.face.EnableWedgeTex();
     648                         LoadGeometry(newMesh, info, refNode.toElement(),materialBindingMap);
     649 #endif
     650                         tri::UpdatePosition<ColladaMesh>::Matrix(newMesh,curTr);
     651                         tri::Append<ColladaMesh,ColladaMesh>::Mesh(m,newMesh);
     652                         QDEBUG("** instance_geometry with url %s (final mesh size %i %i - %i %i)",qPrintable(instGeomNode.attribute("url")),m.vn,m.vert.size(),m.fn,m.face.size());



Build this from ``graphics/meshlab/meshlabplugins/io_collada`` with::

    meshlab-collada-make

Install into the separately built meshlabserver with::

    simon:io_collada blyth$ t meshlab-collada-install
    meshlab-collada-install is a function
    meshlab-collada-install () 
    { 
        cd_func $(env-home)/graphics/meshlab/meshlabplugins/io_collada;
        local dest=../../distrib/plugins/;
        mkdir -p $dest;
        local plug=libio_collada.dylib;
        local target=$dest/$plug;
        rm -rf $target;
        [ -f $plug ] && cp $plug $target
    }



Exploded due to UpdatePosition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contrary to assumption that this was just setting a matrix, it is actually iterating 
over all vertices and multiplying by the matrix.::

     650                         tri::UpdatePosition<ColladaMesh>::Matrix(newMesh,curTr);

Thus the cached ``newMesh`` position keeps changing for each use.
Isolate the cache from its usage with Append, there is no copy ctor::

     636                         ColladaMesh newMesh ;   // avoid exploding the geometry by keeping the cached mesh and newMesh distinct
     637                         tri::Append<ColladaMesh,ColladaMesh>::Mesh(newMesh,*cacheMeshPtr);


::

    ****** LoadPolygonalListMesh (final  mesh size vn 50 vertsize 50 - fn 192 facesize 192)
    **** Loading a Geometry Mesh **** (final   mesh size 50 50 - 192 192)
    ** instance_geometry with url #near-radslab-box-90xc8e73c0 (final mesh size 1246046 1246046 - 2452916 2452916)
    LOG: 0 Opened mesh /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae in 587294 msec
    LOG: 0 All files opened in 594080 msec


Thats better, geometry looks OK, and loading in less than 10 min on ancient laptop::

    In [9]: 594080./1000./60.
    Out[9]: 9.9013333333333335



