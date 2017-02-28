#version 130

uniform sampler2D poolImage;
uniform sampler2D gradImage;
uniform sampler2D convImage;

varying vec2 outUV;


out vec4 gradTex;

vec4 max_val, conv_val, step_val;
void main()
{
 max_val = texture2D(poolImage, outUV);
 conv_val = texture2D(convImage, outUV);
 step_val = step(max_val, conv_val); 

 //pooling Map * reLU Map * gradient Backprop
 gradTex = step_val * sign(conv_val) * texture2D(gradImage, outUV);
}
