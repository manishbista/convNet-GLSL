#version 130

attribute vec3 vertex;
attribute vec3 normal;
attribute vec3 tangent;
attribute vec3 color;
attribute vec2 UV;

uniform float SCREEN_WIDTH;
uniform float SCREEN_HEIGHT;
uniform float TEX_WIDTH;
uniform float TEX_HEIGHT;

uniform mat4 modelViewProjectionMatrix;

varying float left, right, bottom, top;

void main()
{
	gl_Position=modelViewProjectionMatrix*vec4(vertex,1.0);

	right = (2.0 * SCREEN_WIDTH *  UV.x + 1.0) / TEX_WIDTH;
	left = (2.0 * SCREEN_WIDTH *  UV.x) / TEX_WIDTH;
	top = (2.0 * SCREEN_HEIGHT * UV.y) / TEX_HEIGHT;
	bottom = (2.0 * SCREEN_HEIGHT * UV.y + 1.0)/TEX_HEIGHT;
	
}
