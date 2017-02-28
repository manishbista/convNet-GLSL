#version 130

uniform sampler2D grayInputImage;
uniform mat3 kernelMatrix[18];

varying float leftTex, rightTex, topTex, bottomTex, centerXTex, centerYTex;
vec3 colorValueBL, colorValueBC, colorValueBR, colorValueML, colorValueMC, colorValueMR, colorValueTL, colorValueTC, colorValueTR;
vec4 outputColorValue;
mat3 pixelMatrix;

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

//first depth slice
 pixelMatrix = kernelMatrix[0];
 outputColorValue.x = dot(pixelMatrix[0], colorValueTL) + dot(pixelMatrix[1], colorValueTC) + dot(pixelMatrix[2], colorValueTR);

 pixelMatrix = kernelMatrix[1];
 outputColorValue.x = outputColorValue.x + 
 						dot(pixelMatrix[0], colorValueML) + dot(pixelMatrix[1], colorValueMC) + dot(pixelMatrix[2], colorValueMR);

 pixelMatrix = kernelMatrix[2];
 outputColorValue.x = outputColorValue.x + 
 						dot(pixelMatrix[0], colorValueBL) + dot(pixelMatrix[1], colorValueBC) + dot(pixelMatrix[2], colorValueBR);
 
 outputColorValue.x = max(0.0, outputColorValue.x);

//second slice

 pixelMatrix = kernelMatrix[3];
 outputColorValue.y = dot(pixelMatrix[0], colorValueTL) + dot(pixelMatrix[1], colorValueTC) + dot(pixelMatrix[2], colorValueTR);

 pixelMatrix = kernelMatrix[4];
 outputColorValue.y = outputColorValue.y + 
 						dot(pixelMatrix[0], colorValueML) + dot(pixelMatrix[1], colorValueMC) + dot(pixelMatrix[2], colorValueMR);

 pixelMatrix = kernelMatrix[5];
 outputColorValue.y = outputColorValue.y + 
 						dot(pixelMatrix[0], colorValueBL) + dot(pixelMatrix[1], colorValueBC) + dot(pixelMatrix[2], colorValueBR);
 
 outputColorValue.y = max(0.0, outputColorValue.y);


//third slice

 pixelMatrix = kernelMatrix[6];
 outputColorValue.z = dot(pixelMatrix[0], colorValueTL) + dot(pixelMatrix[1], colorValueTC) + dot(pixelMatrix[2], colorValueTR);

 pixelMatrix = kernelMatrix[7];
 outputColorValue.z = outputColorValue.z + 
 						dot(pixelMatrix[0], colorValueML) + dot(pixelMatrix[1], colorValueMC) + dot(pixelMatrix[2], colorValueMR);

 pixelMatrix = kernelMatrix[8];
 outputColorValue.z = outputColorValue.z + 
 						dot(pixelMatrix[0], colorValueBL) + dot(pixelMatrix[1], colorValueBC) + dot(pixelMatrix[2], colorValueBR);
 
 outputColorValue.z = max(0.0, outputColorValue.z);

//save values
 texA = vec4(outputColorValue.x, outputColorValue.y ,outputColorValue.z, 1.0);



//fourth depth slice
 pixelMatrix = kernelMatrix[9];
 outputColorValue.x = dot(pixelMatrix[0], colorValueTL) + dot(pixelMatrix[1], colorValueTC) + dot(pixelMatrix[2], colorValueTR);

 pixelMatrix = kernelMatrix[10];
 outputColorValue.x = outputColorValue.x + 
 						dot(pixelMatrix[0], colorValueML) + dot(pixelMatrix[1], colorValueMC) + dot(pixelMatrix[2], colorValueMR);

 pixelMatrix = kernelMatrix[11];
 outputColorValue.x = outputColorValue.x + 
 						dot(pixelMatrix[0], colorValueBL) + dot(pixelMatrix[1], colorValueBC) + dot(pixelMatrix[2], colorValueBR);
 
 outputColorValue.x = max(0.0, outputColorValue.x);


//fifth slice
 pixelMatrix = kernelMatrix[12];
 outputColorValue.y = dot(pixelMatrix[0], colorValueTL) + dot(pixelMatrix[1], colorValueTC) + dot(pixelMatrix[2], colorValueTR);

 pixelMatrix = kernelMatrix[13];
 outputColorValue.y = outputColorValue.y + 
 						dot(pixelMatrix[0], colorValueML) + dot(pixelMatrix[1], colorValueMC) + dot(pixelMatrix[2], colorValueMR);

 pixelMatrix = kernelMatrix[14];
 outputColorValue.y = outputColorValue.y + 
 						dot(pixelMatrix[0], colorValueBL) + dot(pixelMatrix[1], colorValueBC) + dot(pixelMatrix[2], colorValueBR);
 
 outputColorValue.y = max(0.0, outputColorValue.y);


//sixth slice

 pixelMatrix = kernelMatrix[15];
 outputColorValue.z = dot(pixelMatrix[0], colorValueTL) + dot(pixelMatrix[1], colorValueTC) + dot(pixelMatrix[2], colorValueTR);

 pixelMatrix = kernelMatrix[16];
 outputColorValue.z = outputColorValue.z + 
 						dot(pixelMatrix[0], colorValueML) + dot(pixelMatrix[1], colorValueMC) + dot(pixelMatrix[2], colorValueMR);

 pixelMatrix = kernelMatrix[17];
 outputColorValue.z = outputColorValue.z + 
 						dot(pixelMatrix[0], colorValueBL) + dot(pixelMatrix[1], colorValueBC) + dot(pixelMatrix[2], colorValueBR);
 
 outputColorValue.z = max(0.0, outputColorValue.z);

//save values
 texB = vec4(outputColorValue.x, outputColorValue.y ,outputColorValue.z, 1.0);

}



