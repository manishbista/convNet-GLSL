#version 130
attribute vec3 vertex;
attribute vec2 UV;

varying vec2 outUV;
uniform float SCREEN_WIDTH;
uniform float SCREEN_HEIGHT;
uniform float TEX_WIDTH;
uniform float TEX_HEIGHT;

uniform mat4 modelViewProjectionMatrix;

void main()
{
	gl_Position=modelViewProjectionMatrix*vec4(vertex,1.0);
	outUV.x = UV.x * SCREEN_WIDTH/TEX_WIDTH;
	outUV.y = UV.y * SCREEN_HEIGHT/TEX_HEIGHT;	
}
