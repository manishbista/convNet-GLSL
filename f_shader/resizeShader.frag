#version 130

uniform sampler2D poolImageA;
uniform sampler2D poolImageB;
uniform sampler2D gradImageA;
uniform sampler2D gradImageB;
uniform sampler2D inputImage;

varying vec2 outUV;


out vec4 poolTexA;
out vec4 poolTexB;
out vec4 gradTexA;
out vec4 gradTexB;
out vec4 saveInputImage;

void main()
{

 poolTexA = texture2D(poolImageA, outUV);
 poolTexB = texture2D(poolImageB, outUV);
 gradTexA = texture2D(gradImageA, outUV);
 gradTexB = texture2D(gradImageB, outUV);
 saveInputImage = texture2D(inputImage, outUV);

}
