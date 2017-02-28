#version 130
uniform sampler2D texA;
uniform sampler2D texB;
varying vec2 outUV;

out vec4 poolTexA;
out vec4 poolTexB;
vec4 maxA, maxB;
vec4 vecPoolA, vecPoolB, vecPoolC, vecPoolD;

varying float left, right, bottom, top;

void main()
{

 vecPoolA = texture2D(texA, vec2(left, top));
 vecPoolB = texture2D(texA, vec2(left, bottom));
 vecPoolC = texture2D(texA, vec2(right, top));
 vecPoolD = texture2D(texA, vec2(right, bottom));	

 maxA = max(vecPoolA, vecPoolB);
 maxB = max(vecPoolC, vecPoolD);
 poolTexA = max(maxA, maxB);


//next 4-kernel Images or Depth Slices
 vecPoolA = texture2D(texB, vec2(left, top));
 vecPoolB = texture2D(texB, vec2(left, bottom));
 vecPoolC = texture2D(texB, vec2(right, top));
 vecPoolD = texture2D(texB, vec2(right, bottom));	

 maxA = max(vecPoolA, vecPoolB);
 maxB = max(vecPoolC, vecPoolD);
 poolTexB = max(maxA, maxB);

}
