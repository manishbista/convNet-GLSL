#version 130

uniform sampler2D poolImage;
uniform sampler2D gradImage;
uniform sampler2D sumInputImageA;
uniform sampler2D sumInputImageB;
uniform sampler2D inputImageA;
uniform sampler2D inputImageB;

varying vec2 outUV;

out vec4 poolTex;
out vec4 gradTex;
out vec4 saveInputTopImageA;
out vec4 saveInputTopImageB;

void main()
{

 poolTex = texture2D(poolImage, outUV);
 gradTex = vec4(vec3(texture2D(gradImage, outUV).rgb) * 1000.0, 1.0);
 saveInputTopImageA = vec4(vec3(texture2D(sumInputImageA, outUV)).rgb + vec3(texture2D(inputImageA, outUV)).rgb, 1.0);
 saveInputTopImageB = vec4(vec3(texture2D(sumInputImageB, outUV)).rgb + vec3(texture2D(inputImageB, outUV)).rgb, 1.0);
}
