/**

  * TODO:

    * this fiddle has much more responsive controls that mine, why ? maybe param tuning 

      * http://jsfiddle.net/RnFqz/22/  

    * parent link 
    * form controls, webpy POST such that dont need to remember the query param names but can still use them
    * face highlighting 
    * cull excessive siblings
    * transparency
    * html tree

   * http://belle7.nuu.edu.tw/dae/tree/3154.html

     * the AD is on its side : fixed by changing to Z_UP at DAECopy stage

   * http://localhost/dae/tree/3199.html?cam=0.1,0.1,0.1&wireframe=1

     * inside a PMT 
   
   * http://belle7.nuu.edu.tw/dae/tree/3154.html?cam=0.1,0.1,0.1&anim=1&rotate=0,0.1,0

     * TODO: change to spec axis of rotation and delta ?
       
   * http://belle7.nuu.edu.tw/dae/tree/3154___0.html?cam=0.1,0.1,0.1

     * see nothing here, with camera at center : need to fix double sided 
     * now double sided and sides visble, but curious culling

  * http://belle7.nuu.edu.tw/dae/tree/3154.html?fov=0
  * http://belle7.nuu.edu.tw/dae/tree/2___1.html?cam=0.5,0.5,0.5&fov=0
  
    * frame filling from orthographic ?

   * http://belle7.nuu.edu.tw/dae/tree/3154___0.html?bbcam=1&cam=0,4,1&anim=1&fov=80&rotation=0.0,0.0,0.01
       
     * curious cut, hitting a edge ?

   * http://belle7.nuu.edu.tw/dae/tree/3153___0.html?bbcam=1&cam=0,4,1&anim=1&fov=80&rotation=0.0,0.0,0.01

     * evern wierder cutting 

   * http://localhost/dae/tree/3155___1.html?cam=2,2,2

     * depth is not enough to protect from heavy geometry for some volumes, 
       maybe could cull excessive siblings 

   * http://localhost/dae/tree/2___1.html?cam=0.5,0.5,0.5

   * http://belle7.nuu.edu.tw/dae/tree/0.html?cam=0.1,0.1,0.1&anim=1

     * with double sided on, the view from inside is kinda confusing



*/

THREE_enum_side = { 'double':THREE.DoubleSide, 'front':THREE.FrontSide, 'back':THREE.BackSide };
THREE_enum_bool = { '1':true , '0':false };
THREE_Vector3_fromString = function( s ){
  var xyz = s.split(",");
  var x,y,z ;
  if( xyz.length == 1 ){  
      x = parseFloat(xyz[0]) || 0 ;
      y = x;
      z = x; 
  } else {
      x = parseFloat(xyz[0]) || 0 ;
      y = parseFloat(xyz[1]) || 0 ;
      z = parseFloat(xyz[2]) || 0 ;
   } 
   return new THREE.Vector3(x,y,z) ;
}

DAELOAD_bool_enum = { '1':true , '0':false };
DAELOAD_bool_fromString = function( s ){
   return DAELOAD_bool_enum[s] ;

}

DAELOAD = function(){

        var param ;
        var table ;
        var punit ;
        var renderer, width, height ; 
        var camera ;
        var geometry, material ;
        var mesh, scene ;
        var rotation ;
        var controls ;
        var dae ; 

        function init( _param ){
            param = _param ;
            url = param.url || "../static/models/demo.dae"  ;
            console.log("init loading " + url);
		    var loader = new THREE.ColladaLoader();
		    loader.options.convertUpAxis = true;
		    loader.load( url , handle_load );
        } 

        function handle_load( collada ){
            console.log("handle_load");
		    dae = collada.scene;

            init_renderer(); 
            init_loaded() ;
            init_scene();
            init_camera();
            init_controls_trackball();
            //init_controls_orbit();

            var defaults = { anim:"1" , rotation:"0.01,0.01,0.01" };
            var anim = THREE_enum_bool[param.anim || defaults.anim] ;
            rotation = THREE_Vector3_fromString( param.rotation || defaults.rotation );

            if( anim )
            {
                animate();
            }
            else
            {
                render();
            }
        }


        function param_table_add( table, k, v, link )
        {
            var tr = document.createElement('tr') ;
            var key = document.createElement('td') ;
            var val = document.createElement('td') ;
            key.innerHTML = k ;
            val.innerHTML = ( link ) ? '<a href="' + v + '">' + v + '</a>' : v ;
            tr.appendChild(key);   
            tr.appendChild(val);   
            table.appendChild( tr );
        } 


        function info_div(){
            var info = document.createElement( 'div' );
            info.style.position = 'absolute';
            info.style.top = '10px';
            info.style.width = '100%';
            info.style.textAlign = 'left';

            table = param_table();
            info.appendChild( table );
            return info ;
        }

        function param_table(){
             var table = document.createElement('table') ;
             param_table_add( table, "location", window.location , true );
             for (var k in param) {
                 if (param.hasOwnProperty(k)) {
                     param_table_add( table, k, param[k], k === "url" );
                 }
             }
             return table ;
        }

        function webgl_or_canvas_renderer() {
             var r ;
             if ( Detector.webgl ){
                   r = new THREE.WebGLRenderer() ;
                   param_table_add( table, "renderer", "WebGL",  false );
             } else {
                   r = new THREE.CanvasRenderer();
                   param_table_add( table, "renderer", "Canvas fallback",  false );
             }
             return r ;
        }

        function init_renderer() {
           /*
                ``id``
                           container id
                ``r/renderer``
                           one of auto, webgl, svg, canvas default: auto
                ``clear``
                           clear color

            */

            console.log("init_renderer");
            var defaults = { id:"container" , clear:"dddddd", renderer:"auto" } ;
            var id = param.id || defaults.id ;

            var clear  = parseInt(param.clear || defaults.clear,16) ;  // hex color 
            var rdr    = param.renderer || param.r || defaults.renderer ;  
            rdr = rdr.charAt(0);

            var container = document.getElementById(id);
            width = container.offsetWidth ;
            height = container.offsetHeight ;
            var info = info_div();
            container.appendChild( info );

            switch ( rdr ){
                case 'c': 
                          renderer = new THREE.CanvasRenderer();
                          break;
                case 'w': 
                          renderer = new THREE.WebGLRenderer();
                          break;
                case 's': 
                          renderer = new THREE.SVGRenderer();
                          break;
                case 'a': 
                          renderer = webgl_or_canvas_renderer();
                          break;
            }
            renderer.setSize( width, height );
            renderer.setClearColor( clear , 1 );
            container.appendChild( renderer.domElement );
        }


        function init_loaded(){

            console.log("init_loaded");
            var defaults = { face:"-1" , wireframe:"1" };

            var iface = parseInt(param.face || defaults.face ) ;
            var wireframe = THREE_enum_bool[param.wireframe || param.w || defaults.wireframe ] ;

            var root = dae ;
            var ppv = root.children[0] ; 
            var  lv = ppv.children[0] ; 

            mesh = lv ;           
            geometry = mesh.geometry ; 
            material = mesh.material ; 
            material.wireframe = wireframe ; 


            param_table_add( table, "ppv", ppv.name , false );
            param_table_add( table, "lv" , lv.name  , false );
            param_table_add( table, "geo" , geometry.id  , false );
            param_table_add( table, "mat" , material.name  , false );


            if( iface > -1 )
            {   
                // trying to change a face color http://jsfiddle.net/RnFqz/22/
  
                mesh.dynamic = true ;
                mesh.needsUpdate = true ;

                geometry.dynamic = true ;
                geometry.verticesNeedUpdate = true ;
                geometry.colorsNeedUpdate = true ;

                material.vertexColors = THREE.FaceColors ;

                //var red = new THREE.Color(0xff0000);

                for (var i = 0; i < geometry.faces.length; i++) {
                   var f = geometry.faces[i];
                   f.color.setRGB(Math.random(), Math.random(), Math.random()); 

                  /*
                   var faceIndices = [ 'a', 'b', 'c', 'd' ];
                   n = ( f instanceof THREE.Face3 ) ? 3 : 4;
                   for( var j = 0; j < n; j++ ) {
                        vertexIndex = f[faceIndices[j]];
                        p = geometry.vertices[vertexIndex];
                        param_table_add( table, "f " + i + ' ' + j  , p.x + ' ' + p.y + ' ' + p.z , false );
                   }
                  */

                }

                /* 
                var f = geometry.faces[iface] ; 
                if ( typeof(f) !== "undefined" ){
                   f.color = color;
                }
                */
            }

        } 

        function init_scene(){
            scene = new THREE.Scene();
            instrument_object( mesh , param );
            scene.add( mesh ); 
        }



        function init_camera(){
  /*

         `bb/bbunit=1`   
                 the default of 1 specifies all dimensions in units of the 
                 current root volume bounding box 
                 (maximum absolute extent along any axis) 
                 when using 0, dimensions are in mm 
                 Using mm is difficult, will need to fiddle with parameters to 
                 see anything.

         `n/near=0.1`   
                 camera near plane (the screen)

         `a/far=100`   
                 camera far plane 

         `c/cam=2,2,2`   
                 camera position 

         `l/look=0,0,0`   
                 camera target

         `f/fov=50`
                 vertical field of view in degrees, set to zero for orthographic camera

         `o/orth=10,10,1`   
                 orthographic camera left-right and up-down multiples 

  */

            console.log("init_camera");
            var defaults = { fov:"50", near:"0.1", far:"100", cam:"2,2,2", look:"0,0,0",  bbunit:"1" , orth:"10,10,1" };
 
            var bbunit = THREE_enum_bool[param.bbunit || param.bb || defaults.bbunit ];        // determines the unit of the positions
            var near = parseFloat(param.near || param.n || defaults.near) ;
            var far = parseFloat(param.far   || param.a || defaults.far) ; 
            var cam = THREE_Vector3_fromString( param.cam || param.c || defaults.cam );
            var look = THREE_Vector3_fromString( param.look || param.l || defaults.look );
            var fov = parseFloat(param.fov || param.f || defaults.fov) ;   
            var orth = THREE_Vector3_fromString( param.orth || param.o || defaults.orth );

            if ( bbunit ){
                //
                // when using `bbunit=1` the `cam=2,2,2` and `look=0,0,0` args are 
                // regarded to be in units 
                // of the bounding size obtained from the maximum absolute extent along any axis
                //
                // this allows the camera to positioned inside OR outside an
                // unknown geometry such that on rotating around the geometry should
                // never knock over the camera  
                //
                geometry.computeBoundingBox();
                var bb = geometry.boundingBox  ; 
                punit = Math.max(
                                Math.abs(bb.max.x),Math.abs(bb.min.x), 
                                Math.abs(bb.max.y),Math.abs(bb.min.y), 
                                Math.abs(bb.max.z),Math.abs(bb.min.z)
                             );
                param_table_add( table, "bbx",  bb.min.x + ' ' + bb.max.x , false );
                param_table_add( table, "bby",  bb.min.y + ' ' + bb.max.y , false );
                param_table_add( table, "bbz",  bb.min.z + ' ' + bb.max.z , false );
            }
            else
            {
                punit = 1.0 ; 
            }


            cam.multiplyScalar( punit );
            look.multiplyScalar( punit );
            near = near * punit ;
            far = far * punit ;

            param_table_add( table, "punit", punit , false );
            param_table_add( table, "fov", fov , false );
            param_table_add( table, "near", near , false );
            param_table_add( table, "far",  far , false );
            param_table_add( table, "cam",   cam.x + ' ' + cam.y + ' ' + cam.z , false );
            param_table_add( table, "look",  look.x + ' ' + look.y + ' ' + look.z , false );

            if ( fov == 0 )
            {
               // only near far are punit scaled ???
                var right = orth.x * width * 0.5  ; 
                var top   = orth.y * height * 0.5 ; 
                camera = new THREE.OrthographicCamera( -right, right, top, -top, near, far );
            }
            else
            {
                camera = new THREE.PerspectiveCamera( fov, width / height, near, far );
            }

            camera.position.copy(cam);
            camera.lookAt(look);
        }


        function init_controls_orbit(){

             var defaults = { zoomspeed:"1" };
             var zoomspeed = parseFloat( param.zoomspeed || defaults.zoomspeed ) ;   
             param_table_add( table, "zoomspeed",  zoomspeed , false );

             controls = new THREE.OrbitControls( camera );
             controls.zoomSpeed = zoomspeed ;  // this is actually dollying, no fov change ?

           /*
             controls.rotateSpeed = 1.0;
             controls.zoomSpeed = 1.0;  // this is actually dollying, no fov change ?
             controls.panSpeed = 0.8;
             controls.staticMoving = true;
             controls.dynamicDampingFactor = 0.3;
            */

             controls.addEventListener( 'change', render );
        }

        function init_controls_trackball(){

             /*
                  http://jsfiddle.net/RnFqz/22/

             */

             controls = new THREE.TrackballControls(camera);
             controls.rotateSpeed = 1.0;
             controls.zoomSpeed = 1.2;
             controls.panSpeed = 0.2;
             controls.noZoom = false;
             controls.noPan = false;
             controls.staticMoving = true;
             controls.dynamicDampingFactor = 0.3;
             controls.keys = [65, 83, 68];
        } 


        function init_cube( param ){

            var defaults = { size:200, color:0xff0000, wireframe:"1", side:"double", obj:"0,0,0" };

            var obj = THREE_Vector3_fromString( param.obj || defaults.obj );
            var size = param.size || defaults.size ;
            var color = param.color || defaults.color
            var wireframe = THREE_enum_bool[param.wireframe || defaults.wireframe] ;
            var side = THREE_enum_side[param.side || defaults.side] ;  

            geometry = new THREE.CubeGeometry( size, size, size );
            material = new THREE.MeshBasicMaterial( { color: color, wireframe: wireframe } );
            material.side = side ; 
            mesh = new THREE.Mesh( geometry, material );  // fallback to cube
            mesh.position.copy(obj);
        }

        function instrument_object( obj , param ){

            var defaults = { axes:"1" , bbox:"1" };

            var axes = THREE_enum_bool[param.axes || defaults.axes] ;
            var bbox = THREE_enum_bool[param.bbox || defaults.bbox] ;

            if( axes ){

               if ( obj.geometry.boundingBox === null ) {
                    obj.geometry.computeBoundingBox();
               }
               var bb = obj.geometry.boundingBox  ; 
               var size = Math.max( 
                          Math.abs(bb.min.x), Math.abs(bb.max.x), 
                          Math.abs(bb.min.y), Math.abs(bb.max.y), 
                          Math.abs(bb.min.z), Math.abs(bb.max.z)
                          );                // cannot use punit as that is 1 if not using bbunit 
                obj.add(new THREE.DoubleAxisHelper( size * 1.2 ));
                obj.add(new THREE.DoubleAxisHelper( size * -1.2 ));
            }

            if( bbox ){
                obj.add(new THREE.BoxHelper(obj)) ; 
            }
        }


        function animate() {
            requestAnimationFrame( animate );
            render();
            controls.update();
        }

        function render() {
            renderer.render( scene, camera );
        }

        return {
            init : init,
            animate : animate, 
            param : param 
        };


}();

