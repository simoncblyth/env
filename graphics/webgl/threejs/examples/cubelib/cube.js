/**

    http://localhost:8000/cubelib/cubelib.html?wireframe=1&side=double&cam=0,0,500&obj=0,0,0&look=300,0,0

*/

THREE_Vector3_fromString = function( s ){
   var xyz = s.split(",");
   x = parseFloat(xyz[0]) || 0 ;
   y = parseFloat(xyz[1]) || 0 ;
   z = parseFloat(xyz[2]) || 0 ;
   return new THREE.Vector3(x,y,z) ;
}


CUBE = function(){

        var param ;
        var camera, scene, renderer;
        var geometry, material, mesh;

        var enum_side = { 'double':THREE.DoubleSide, 'front':THREE.FrontSide, 'back':THREE.BackSide };
        var enum_wireframe = { '1':true , '0':false };

        function init(_param) {
            param = _param ; 
            var defaults = { id:"container", fov:75, near:1, far:10000, cam:"0,0,1000", look:"0,0,0", obj:"0,0,0", size:200, color:0xff0000, wireframe:"1", side:"double" };

            var id = param.id || defaults.id ;
            var fov = param.fov || defaults.fov ;
            var near = param.near || defaults.near ;
            var far = param.far || defaults.far ; 
            var size = param.size || defaults.size ;
            var color = param.color || defaults.color
            var wireframe = enum_wireframe[param.wireframe || defaults.wireframe] ;
            var side = enum_side[param.side || defaults.side] ;  

            var cam = THREE_Vector3_fromString(param.cam || defaults.cam );
            var look = THREE_Vector3_fromString(param.look || defaults.look );
            var obj = THREE_Vector3_fromString(param.obj || defaults.obj );

            var container = document.getElementById(id);
            var width = container.offsetWidth ;
            var height = container.offsetHeight ;

            renderer = new THREE.CanvasRenderer();
            renderer.setSize( width, height );
            container.appendChild( renderer.domElement );

            if ( fov == 0 )
            {
                var right = width / 2 ; 
                var left = - right ; 
                var top = height  ; 
                var bottom = - top ; 
                camera = new THREE.OrthographicCamera( left, right, top, bottom, near, far );
            }
            else
            {
                camera = new THREE.PerspectiveCamera( fov, width / height, near, far );
            }
            camera.position.copy(cam);
            camera.lookAt(look);


            geometry = new THREE.CubeGeometry( size, size, size );
            material = new THREE.MeshBasicMaterial( { color: color, wireframe: wireframe } );
            material.side = side ; 

            mesh = new THREE.Mesh( geometry, material );
            mesh.position.copy(obj);

            scene = new THREE.Scene();
            scene.add( mesh );
        }

        function animate() {
            requestAnimationFrame( animate );
            mesh.rotation.x += 0.01;
            mesh.rotation.y += 0.02;
            renderer.render( scene, camera );
        }

        return {
            init : init,
            animate : animate
        };


}();

