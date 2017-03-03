#version 130

uniform sampler2D grayInputImage;
uniform sampler2D saveInputImage;
uniform sampler2D gradientImageA;
uniform sampler2D gradientImageB;
uniform mat3 kernelMatrix[18];

varying float leftTex, rightTex, topTex, bottomTex, centerXTex, centerYTex;
vec3 colorValueBL, colorValueBC, colorValueBR, colorValueML, colorValueMC, colorValueMR, colorValueTL, colorValueTC, colorValueTR;
vec3 colorValueSBL, colorValueSBC, colorValueSBR, colorValueSML, colorValueSMC, colorValueSMR, colorValueSTL, colorValueSTC, colorValueSTR;
vec4 outputColorValue;
mat3 pixelMatrix;
vec4 gradTextureA, gradTextureB;
vec3 pixelVector;


out vec4 texA;
out vec4 texB;

void main()
{

//RGB values in 9 spatial locations = 9X3 vectors

 colorValueBL = texture2D(grayInputImage, vec2(leftTex, bottomTex)).rgb;
 colorValueBC = texture2D(grayInputImage, vec2(centerXTex, bottomTex)).rgb;
 colorValueBR = texture2D(grayInputImage, vec2(rightTex, bottomTex)).rgb;

 colorValueML = texture2D(grayInputImage, vec2(leftTex, centerYTex)).rgb;
 colorValueMC = texture2D(grayInputImage, vec2(centerXTex, centerYTex)).rgb;
 colorValueMR = texture2D(grayInputImage, vec2(rightTex, centerYTex)).rgb;

 colorValueTL = texture2D(grayInputImage, vec2(leftTex, topTex)).rgb;
 colorValueTC = texture2D(grayInputImage, vec2(centerXTex, topTex)).rgb;
 colorValueTR = texture2D(grayInputImage, vec2(rightTex, topTex)).rgb;

 gradTextureA = texture2D(gradientImageA, vec2(centerXTex, centerYTex));
 gradTextureB = texture2D(gradientImageB, vec2(centerXTex, centerYTex));

 colorValueSBL = texture2D(saveInputImage, vec2(leftTex, bottomTex)).rgb;
 colorValueSBC = texture2D(saveInputImage, vec2(centerXTex, bottomTex)).rgb;
 colorValueSBR = texture2D(saveInputImage, vec2(rightTex, bottomTex)).rgb;

 colorValueSML = texture2D(saveInputImage, vec2(leftTex, centerYTex)).rgb;
 colorValueSMC = texture2D(saveInputImage, vec2(centerXTex, centerYTex)).rgb;
 colorValueSMR = texture2D(saveInputImage, vec2(rightTex, centerYTex)).rgb;

 colorValueSTL = texture2D(saveInputImage, vec2(leftTex, topTex)).rgb;
 colorValueSTC = texture2D(saveInputImage, vec2(centerXTex, topTex)).rgb;
 colorValueSTR = texture2D(saveInputImage, vec2(rightTex, topTex)).rgb;

//first depth slice
 pixelMatrix = kernelMatrix[0];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.x = dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[1];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[2];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));
 
 outputColorValue.x = max(0.0, outputColorValue.x);

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
 
 outputColorValue.y = max(0.0, outputColorValue.y);


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
 
 outputColorValue.z = max(0.0, outputColorValue.z);

//save values
 texA = vec4(outputColorValue.x, outputColorValue.y ,outputColorValue.z, 1.0);



//fourth depth slice
 pixelMatrix = kernelMatrix[9];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.x = dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[10];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[11];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.x = outputColorValue.x + dot(pixelVector, vec3(1.0));
 
 outputColorValue.x = max(0.0, outputColorValue.x);


//fifth slice
 pixelMatrix = kernelMatrix[12];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.y = dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[13];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.y = outputColorValue.y + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[14];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.y = outputColorValue.y + dot(pixelVector, vec3(1.0));
 
 outputColorValue.y = max(0.0, outputColorValue.y);


//sixth slice

 pixelMatrix = kernelMatrix[15];
 pixelVector = pixelMatrix[0] * colorValueTL + pixelMatrix[1] * colorValueTC + pixelMatrix[2] * colorValueTR;
 outputColorValue.z = dot(pixelVector, vec3(1.0)); 

 pixelMatrix = kernelMatrix[16];
 pixelVector = pixelMatrix[0] * colorValueML + pixelMatrix[1] * colorValueMC + pixelMatrix[2] * colorValueMR;
 outputColorValue.z = outputColorValue.z + dot(pixelVector, vec3(1.0));

 pixelMatrix = kernelMatrix[17];
 pixelVector = pixelMatrix[0] * colorValueBL + pixelMatrix[1] * colorValueBC + pixelMatrix[2] * colorValueBR;
 outputColorValue.z = outputColorValue.z + dot(pixelVector, vec3(1.0));
 
 outputColorValue.z = max(0.0, outputColorValue.z);

//save values
 texB = vec4(outputColorValue.x, outputColorValue.y ,outputColorValue.z, 1.0);

}



