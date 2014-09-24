/**
 * @author sroucheray / http://sroucheray.org/
 * @author mrdoob / http://mrdoob.com/
 * 
 * SCB dashed axes when size if negative
 */

THREE.DoubleAxisHelper = function ( size ) {

	size = size || 1;

	var geometry = new THREE.Geometry();

	geometry.vertices.push(
		new THREE.Vector3(), new THREE.Vector3( size, 0, 0 ),
		new THREE.Vector3(), new THREE.Vector3( 0, size, 0 ),
		new THREE.Vector3(), new THREE.Vector3( 0, 0, size )
	);

	geometry.colors.push(
		new THREE.Color( 0xff0000 ), new THREE.Color( 0xffaa00 ),
		new THREE.Color( 0x00ff00 ), new THREE.Color( 0xaaff00 ),
		new THREE.Color( 0x0000ff ), new THREE.Color( 0x00aaff )
	);

   
	var material = ( size > 0 ) ? 
          new THREE.LineBasicMaterial(  { vertexColors: THREE.VertexColors } ) :
          new THREE.LineDashedMaterial( { vertexColors: THREE.VertexColors , dashSize: size/50, gapSize: size/50 } ) ;

	THREE.Line.call( this, geometry, material, THREE.LinePieces );

};

THREE.DoubleAxisHelper.prototype = Object.create( THREE.Line.prototype );
