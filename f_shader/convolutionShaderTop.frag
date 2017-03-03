#version 130

uniform sampler2D firstInputImage;
uniform sampler2D secondInputImage;
uniform sampler2D saveInputImageA;
uniform sampler2D saveInputImageB;
uniform sampler2D gradientImage;
uniform mat3 kernelMatrix[18];

varying float leftTex, rightTex, topTex, bottomTex, centerXTex, centerYTex;
vec3 colorValueBL, colorValueBC, colorValueBR, colorValueML, colorValueMC, colorValueMR, colorValueTL, colorValueTC, colorValueTR;
vec4 outputColorValue;
mat3 pixelMatrix;
vec4 gradTexture;
vec3 pixelVector;
//color values of saved Texture Image which is of previous iteration, used for updating weights
vec3 colorValueSBL, colorValueSBC, colorValueSBR, colorValueSML, colorValueSMC, colorValueSMR, colorValueSTL, colorValueSTC, colorValueSTR;

out vec4 texA;


void main()
{

//RGB values in 9 spatial locations = 9X3 vectors

 colorValueBL = texture2D(firstInputImage, vec2(leftTex, bottomTex)).rgb;
 colorValueBC = texture2D(firstInputImage, vec2(centerXTex, bottomTex)).rgb;
 colorValueBR = texture2D(firstInputImage, vec2(rightTex, bottomTex)).rgb;

 colorValueML = texture2D(firstInputImage, vec2(leftTex, centerYTex)).rgb;
 colorValueMC = texture2D(firstInputImage, vec2(centerXTex, centerYTex)).rgb;
 colorValueMR = texture2D(firstInputImage, vec2(rightTex, centerYTex)).rgb;

 colorValueTL = texture2D(firstInputImage, vec2(leftTex, topTex)).rgb;
 colorValueTC = texture2D(firstInputImage, vec2(centerXTex, topTex)).rgb;
 colorValueTR = texture2D(firstInputImage, vec2(rightTex, topTex)).rgb;

 gradTexture = texture2D(gradientImage, vec2(centerXTex, centerYTex));

 colorValueSBL = texture2D(saveInputImageA, vec2(leftTex, bottomTex)).rgb;
 colorValueSBC = texture2D(saveInputImageA, vec2(centerXTex, bottomTex)).rgb;
 colorValueSBR = texture2D(saveInputImageA, vec2(rightTex, bottomTex)).rgb;

 colorValueSML = texture2D(saveInputImageA, vec2(leftTex, centerYTex)).rgb;
 colorValueSMC = texture2D(saveInputImageA, vec2(centerXTex, centerYTex)).rgb;
 colorValueSMR = texture2D(saveInputImageA, vec2(rightTex, centerYTex)).rgb;

 colorValueSTL = texture2D(saveInputImageA, vec2(leftTex, topTex)).rgb;
 colorValueSTC = texture2D(saveInputImageA, vec2(centerXTex, topTex)).rgb;
 colorValueSTR = texture2D(saveInputImageA, vec2(rightTex, topTex)).rgb;

//first depth slice
 pixelMatrix = kernelMatrix[0];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.x =  dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[1];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[2];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));

//second slice

 pixelMatrix = kernelMatrix[3];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.y = dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[4];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.y = outputColorValue.y + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[5];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.y = outputColorValue.y + dot(pixelVector, vec3(1.0));

//third slice

 pixelMatrix = kernelMatrix[6];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.z = dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[7];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.z = outputColorValue.z + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[8];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.z = outputColorValue.z + dot(pixelVector, vec3(1.0));
 						
 						
 colorValueBL = texture2D(secondInputImage, vec2(leftTex, bottomTex)).rgb;
 colorValueBC = texture2D(secondInputImage, vec2(centerXTex, bottomTex)).rgb;
 colorValueBR = texture2D(secondInputImage, vec2(rightTex, bottomTex)).rgb;

 colorValueML = texture2D(secondInputImage, vec2(leftTex, centerYTex)).rgb;
 colorValueMC = texture2D(secondInputImage, vec2(centerXTex, centerYTex)).rgb;
 colorValueMR = texture2D(secondInputImage, vec2(rightTex, centerYTex)).rgb;

 colorValueTL = texture2D(secondInputImage, vec2(leftTex, topTex)).rgb;
 colorValueTC = texture2D(secondInputImage, vec2(centerXTex, topTex)).rgb;
 colorValueTR = texture2D(secondInputImage, vec2(rightTex, topTex)).rgb;

 colorValueSBL = texture2D(saveInputImageB, vec2(leftTex, bottomTex)).rgb;
 colorValueSBC = texture2D(saveInputImageB, vec2(centerXTex, bottomTex)).rgb;
 colorValueSBR = texture2D(saveInputImageB, vec2(rightTex, bottomTex)).rgb;

 colorValueSML = texture2D(saveInputImageB, vec2(leftTex, centerYTex)).rgb;
 colorValueSMC = texture2D(saveInputImageB, vec2(centerXTex, centerYTex)).rgb;
 colorValueSMR = texture2D(saveInputImageB, vec2(rightTex, centerYTex)).rgb;

 colorValueSTL = texture2D(saveInputImageB, vec2(leftTex, topTex)).rgb;
 colorValueSTC = texture2D(saveInputImageB, vec2(centerXTex, topTex)).rgb;
 colorValueSTR = texture2D(saveInputImageB, vec2(rightTex, topTex)).rgb;

//first depth slice
 pixelMatrix = kernelMatrix[9];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.x =  outputColorValue.x + dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[10];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[11];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));
 
 outputColorValue.x = max(0.0, outputColorValue.x);


//second slice
 pixelMatrix = kernelMatrix[12];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.y = outputColorValue.y + dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[13];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.y = outputColorValue.y + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[14];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.y = outputColorValue.y + dot(pixelVector, vec3(1.0)); 

 outputColorValue.y = max(0.0, outputColorValue.y);


//third slice

 pixelMatrix = kernelMatrix[15];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.z = outputColorValue.z + dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[16];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.z = outputColorValue.z + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[17];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.z = outputColorValue.z + dot(pixelVector, vec3(1.0));
 
 outputColorValue.z = max(0.0, outputColorValue.z);

//save values
 texA = vec4(outputColorValue.x, outputColorValue.y ,outputColorValue.z, 1.0);

}



