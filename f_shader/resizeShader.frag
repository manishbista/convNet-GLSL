#version 130

uniform sampler2D poolImageA;
uniform sampler2D poolImageB;
uniform sampler2D gradImageA;
uniform sampler2D gradImageB;

varying vec2 outUV;


out vec4 poolTexA;
out vec4 poolTexB;
out vec4 gradTexA;
out vec4 gradTexB;

void main()
{

 poolTexA = texture2D(poolImageA, outUV);
 poolTexB = texture2D(poolImageB, outUV);
 gradTexA = texture2D(gradImageA, outUV);
 gradTexB = texture2D(gradImageB, outUV);
}
