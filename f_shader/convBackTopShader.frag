#version 130

uniform sampler2D inputImage;
uniform mat3 kernelMatrix[18];
varying float leftTex, rightTex, topTex, bottomTex, centerXTex, centerYTex;

vec3 texBL, texBC, texBR, texML, texMC, texMR, texTL, texTC, texTR;
vec3 texA;
vec3 texB;

out vec4 convBackTexA;
out vec4 convBackTexB;


mat3 pixelMatrix;

void main()
{

 texBL = texture2D(inputImage, vec2(leftTex, bottomTex)).rgb;
 texBC = texture2D(inputImage, vec2(centerXTex, bottomTex)).rgb;
 texBR = texture2D(inputImage, vec2(rightTex, bottomTex)).rgb;

 texML = texture2D(inputImage, vec2(leftTex, centerYTex)).rgb;
 texMC = texture2D(inputImage, vec2(centerXTex, centerYTex)).rgb;
 texMR = texture2D(inputImage, vec2(rightTex, centerYTex)).rgb;

 texTL = texture2D(inputImage, vec2(leftTex, topTex)).rgb;
 texTC = texture2D(inputImage, vec2(centerXTex, topTex)).rgb;
 texTR = texture2D(inputImage, vec2(rightTex, topTex)).rgb;


//first depth slice
 texA = vec3(0.0);
 pixelMatrix = kernelMatrix[2];
 texA = texA + pixelMatrix[2] * texTL.r + pixelMatrix[1] * texTC.r + pixelMatrix[0] * texTR.r;

 pixelMatrix = kernelMatrix[1];
 texA = texA + pixelMatrix[2] * texML.r + pixelMatrix[1] * texMC.r + pixelMatrix[0] * texMR.r;

 pixelMatrix = kernelMatrix[0];
 texA = texA + pixelMatrix[2] * texBL.r + pixelMatrix[1] * texBC.r + pixelMatrix[0] * texBR.r;


//first depth slice
 texB = vec3(0.0);
 pixelMatrix = kernelMatrix[11];
 texB = texB + pixelMatrix[2] * texTL.r + pixelMatrix[1] * texTC.r + pixelMatrix[0] * texTR.r;

 pixelMatrix = kernelMatrix[10];
 texB = texB + pixelMatrix[2] * texML.r + pixelMatrix[1] * texMC.r + pixelMatrix[0] * texMR.r;

 pixelMatrix = kernelMatrix[9];
 texB = texB + pixelMatrix[2] * texBL.r + pixelMatrix[1] * texBC.r + pixelMatrix[0] * texBL.r;

//second depth slice
 pixelMatrix = kernelMatrix[5];
 texA = texA + pixelMatrix[2] * texTL.g + pixelMatrix[1] * texTC.g + pixelMatrix[0] * texTR.g;

 pixelMatrix = kernelMatrix[4];
 texA = texA + pixelMatrix[2] * texML.g + pixelMatrix[1] * texMC.g + pixelMatrix[0] * texMR.g;

 pixelMatrix = kernelMatrix[3];
 texA = texA + pixelMatrix[2] * texBL.g + pixelMatrix[1] * texBC.g + pixelMatrix[0] * texBR.g;


//second depth slice
 pixelMatrix = kernelMatrix[14];
 texB = texB + pixelMatrix[2] * texTL.g + pixelMatrix[1] * texTC.g + pixelMatrix[0] * texTR.g;

 pixelMatrix = kernelMatrix[13];
 texB = texB + pixelMatrix[2] * texML.g + pixelMatrix[1] * texMC.g + pixelMatrix[0] * texMR.g;

 pixelMatrix = kernelMatrix[12];
 texB = texB + pixelMatrix[2] * texBL.g + pixelMatrix[1] * texBC.g + pixelMatrix[0] * texBL.g;


//third depth slice
 pixelMatrix = kernelMatrix[8];
 texA = texA + pixelMatrix[2] * texTL.b + pixelMatrix[1] * texTC.b + pixelMatrix[0] * texTR.b;

 pixelMatrix = kernelMatrix[7];
 texA = texA + pixelMatrix[2] * texML.b + pixelMatrix[1] * texMC.b + pixelMatrix[0] * texMR.b;

 pixelMatrix = kernelMatrix[6];
 texA = texA + pixelMatrix[2] * texBL.b + pixelMatrix[1] * texBC.b + pixelMatrix[0] * texBR.b;


//third depth slice
 pixelMatrix = kernelMatrix[17];
 texB = texB + pixelMatrix[2] * texTL.b + pixelMatrix[1] * texTC.b + pixelMatrix[0] * texTR.b;

 pixelMatrix = kernelMatrix[16];
 texB = texB + pixelMatrix[2] * texML.b + pixelMatrix[1] * texMC.b + pixelMatrix[0] * texMR.b;

 pixelMatrix = kernelMatrix[15];
 texB = texB + pixelMatrix[2] * texBL.b + pixelMatrix[1] * texBC.b + pixelMatrix[0] * texBL.b;

 convBackTexA = vec4(texA, 1.0);
 convBackTexB = vec4(texB, 1.0);
}





