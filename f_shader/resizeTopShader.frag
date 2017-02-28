#version 130

uniform sampler2D poolImage;
uniform sampler2D gradImage;

varying vec2 outUV;


out vec4 poolTex;
out vec4 gradTex;

void main()
{

 poolTex = texture2D(poolImage, outUV);
 gradTex = vec4(vec3(texture2D(gradImage, outUV).rgb) * 1000.0, 1.0);
}
