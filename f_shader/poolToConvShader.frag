#version 130

uniform sampler2D poolImageA;
uniform sampler2D poolImageB;
uniform sampler2D gradImageA;
uniform sampler2D gradImageB;
uniform sampler2D convImageA;
uniform sampler2D convImageB;

varying vec2 outUV;


out vec4 gradientTexA;
out vec4 gradientTexB;

vec4 max_val, conv_val, step_val;
void main()
{
 max_val = texture2D(poolImageA, outUV);
 conv_val = texture2D(convImageA, outUV);
 step_val = step(max_val, conv_val); 

 //pooling Map * reLU Map * gradient Backprop
 gradientTexA = step_val * sign(conv_val) * texture2D(gradImageA, outUV);

 max_val = texture2D(poolImageB, outUV);
 conv_val = texture2D(convImageB, outUV);
 step_val = step(max_val, conv_val); 

 //pooling Map * reLU Map * gradient Backprop
 gradientTexB = step_val * sign(conv_val) * texture2D(gradImageB, outUV);
}
