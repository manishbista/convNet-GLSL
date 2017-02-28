#version 130
uniform sampler2D texA;

out vec4 poolTex;
out vec4 poolMap;
vec4 maxA, maxB;
vec4 vecPoolA, vecPoolB, vecPoolC, vecPoolD;

varying float left, right, bottom, top;
varying vec2 outUV;

void main()
{

 vecPoolA = texture2D(texA, vec2(left, top));
 vecPoolB = texture2D(texA, vec2(left, bottom));
 vecPoolC = texture2D(texA, vec2(right, top));
 vecPoolD = texture2D(texA, vec2(right, bottom));	

 maxA = max(vecPoolA, vecPoolB);
 maxB = max(vecPoolC, vecPoolD);
 poolTex = max(maxA, maxB);
}
