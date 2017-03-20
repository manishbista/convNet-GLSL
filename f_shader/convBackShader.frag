#version 130

uniform sampler2D firstInputImage;
uniform sampler2D secondInputImage;
uniform sampler2D saveInputImage;
uniform mat3 kernelMatrix[18];
varying float leftTex, rightTex, topTex, bottomTex, centerXTex, centerYTex;

vec3 texBL, texBC, texBR, texML, texMC, texMR, texTL, texTC, texTR;
vec3 texA;

out vec4 convBackTex;
out vec4 sumImage;

mat3 pixelMatrix;

void main()
{

 texBL = texture2D(firstInputImage, vec2(leftTex, bottomTex)).rgb;
 texBC = texture2D(firstInputImage, vec2(centerXTex, bottomTex)).rgb;
 texBR = texture2D(firstInputImage, vec2(rightTex, bottomTex)).rgb;

 texML = texture2D(firstInputImage, vec2(leftTex, centerYTex)).rgb;
 texMC = texture2D(firstInputImage, vec2(centerXTex, centerYTex)).rgb;
 texMR = texture2D(firstInputImage, vec2(rightTex, centerYTex)).rgb;

 texTL = texture2D(firstInputImage, vec2(leftTex, topTex)).rgb;
 texTC = texture2D(firstInputImage, vec2(centerXTex, topTex)).rgb;
 texTR = texture2D(firstInputImage, vec2(rightTex, topTex)).rgb;


 texA = vec3(0.0);
 pixelMatrix = kernelMatrix[2];
 texA = texA + pixelMatrix[2] * texTL.r + pixelMatrix[1] * texTC.r + pixelMatrix[0] * texTR.r;

 pixelMatrix = kernelMatrix[1];
 texA = texA + pixelMatrix[2] * texML.r + pixelMatrix[1] * texMC.r + pixelMatrix[0] * texMR.r;

 pixelMatrix = kernelMatrix[0];
 texA = texA + pixelMatrix[2] * texBL.r + pixelMatrix[1] * texBC.r + pixelMatrix[0] * texBR.r;


 pixelMatrix = kernelMatrix[5];
 texA = texA + pixelMatrix[2] * texTL.g + pixelMatrix[1] * texTC.g + pixelMatrix[0] * texTR.g;

 pixelMatrix = kernelMatrix[4];
 texA = texA + pixelMatrix[2] * texML.g + pixelMatrix[1] * texMC.g + pixelMatrix[0] * texMR.g;

 pixelMatrix = kernelMatrix[3];
 texA = texA + pixelMatrix[2] * texBL.g + pixelMatrix[1] * texBC.g + pixelMatrix[0] * texBR.g;


 pixelMatrix = kernelMatrix[8];
 texA = texA + pixelMatrix[2] * texTL.b + pixelMatrix[1] * texTC.b + pixelMatrix[0] * texTR.b;

 pixelMatrix = kernelMatrix[7];
 texA = texA + pixelMatrix[2] * texML.b + pixelMatrix[1] * texMC.b + pixelMatrix[0] * texMR.b;

 pixelMatrix = kernelMatrix[6];
 texA = texA + pixelMatrix[2] * texBL.b + pixelMatrix[1] * texBC.b + pixelMatrix[0] * texBR.b;


 texBL = texture2D(secondInputImage, vec2(leftTex, bottomTex)).rgb;
 texBC = texture2D(secondInputImage, vec2(centerXTex, bottomTex)).rgb;
 texBR = texture2D(secondInputImage, vec2(rightTex, bottomTex)).rgb;

 texML = texture2D(secondInputImage, vec2(leftTex, centerYTex)).rgb;
 texMC = texture2D(secondInputImage, vec2(centerXTex, centerYTex)).rgb;
 texMR = texture2D(secondInputImage, vec2(rightTex, centerYTex)).rgb;

 texTL = texture2D(secondInputImage, vec2(leftTex, topTex)).rgb;
 texTC = texture2D(secondInputImage, vec2(centerXTex, topTex)).rgb;
 texTR = texture2D(secondInputImage, vec2(rightTex, topTex)).rgb;

//first depth slice
 pixelMatrix = kernelMatrix[11];
 texA = texA + pixelMatrix[2] * texTL.r + pixelMatrix[1] * texTC.r + pixelMatrix[0] * texTR.r;

 pixelMatrix = kernelMatrix[10];
 texA = texA + pixelMatrix[2] * texML.r + pixelMatrix[1] * texMC.r + pixelMatrix[0] * texMR.r;

 pixelMatrix = kernelMatrix[9];
 texA = texA + pixelMatrix[2] * texBL.r + pixelMatrix[1] * texBC.r + pixelMatrix[0] * texBL.r;


//second depth slice
 pixelMatrix = kernelMatrix[14];
 texA = texA + pixelMatrix[2] * texTL.g + pixelMatrix[1] * texTC.g + pixelMatrix[0] * texTR.g;

 pixelMatrix = kernelMatrix[13];
 texA = texA + pixelMatrix[2] * texML.g + pixelMatrix[1] * texMC.g + pixelMatrix[0] * texMR.g;

 pixelMatrix = kernelMatrix[12];
 texA = texA + pixelMatrix[2] * texBL.g + pixelMatrix[1] * texBC.g + pixelMatrix[0] * texBL.g;


//third depth slice
 pixelMatrix = kernelMatrix[17];
 texA = texA + pixelMatrix[2] * texTL.b + pixelMatrix[1] * texTC.b + pixelMatrix[0] * texTR.b;

 pixelMatrix = kernelMatrix[16];
 texA = texA + pixelMatrix[2] * texML.b + pixelMatrix[1] * texMC.b + pixelMatrix[0] * texMR.b;

 pixelMatrix = kernelMatrix[15];
 texA = texA + pixelMatrix[2] * texBL.b + pixelMatrix[1] * texBC.b + pixelMatrix[0] * texBL.b;

 convBackTex = vec4(texA, 1.0);

 sumImage = texture2D(saveInputImage, vec2(centerXTex, centerYTex));
}





