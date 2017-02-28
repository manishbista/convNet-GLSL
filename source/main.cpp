#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>
#include "../header/meshLoader.h"
#include "../header/shader.h"
#include "../header/matrices.h"
#include <iostream>
#include "../header/mnist.h"
#include "../header/mlp.h"
#include <ctime>


using namespace std;

float SCREEN_WIDTH = 640.0;
float SCREEN_HEIGHT = 480.0;
SDL_Window* gWindow;
SDL_Surface* gScreenSurface;
matrices pipeline;
meshLoader* scene;
mesh* quad;
mesh* quadInverted;

shader* convTopShades;
shader* convShades;
shader* poolingTopShades;
shader* poolingShades;
shader* resizeTopShades;
shader* displayShades;
shader* poolToConvTopShades;
shader* convBackTopShades;
shader* poolToConvShades;
shader* resizeShades;
shader* convBackShades;

idxFileReader* digitSet;
unsigned char* digitImage;
unsigned int digitTexture;

unsigned int colorFBO, colorImage;
unsigned int convFBO, convImageA, convImageB;
unsigned int convTopFBO, convTopImage;
unsigned int poolingFBO, poolingImageA, poolingImageB;
unsigned int poolingTopFBO, poolingTopImage;

unsigned int resizeTopFBO, resizeTopPoolImage, resizeTopGradientImage;
unsigned int poolToConvTopFBO, poolToConvTopImage;
unsigned int convBackTopFBO, convBackTopImageA, convBackTopImageB;
unsigned int poolToConvFBO, poolToConvImageA, poolToConvImageB;
unsigned int convBackFBO, convBackImage;

unsigned int resizeFBO, resizePoolImageA, resizePoolImageB, resizeGradientImageA, resizeGradientImageB;

int fbStatus;

glm::mat3 kernelMatrix;
glm::mat3 kernelMatrix_1[18];
glm::mat3 kernelMatrix_2[18];
float *convPixelBuffer;
float *labelBuffer;
float *convGradientBuffer;
unsigned int gradientTexture;

float testBuffer[14 * 14 * 4];

unsigned int createTexture(int w,int h, bool isFloatTex = false);
unsigned int createTexture(int w, int h, unsigned char *pixels);
unsigned int createTexture(int w, int h, float *pixels);
void updateTextureContent(unsigned int textureId, int w, int h, unsigned char* pixels);
void updateTextureContent(unsigned int textureId, int w, int h, float* pixels);
unsigned int createRGBTexture(int w, int h);


int imgCount;
int imageID = 0;
int label;
int prevImageID = -1;

mlp* mlpObject;
void init()
{
	pipeline.matrixMode(PROJECTION_MATRIX);
	pipeline.loadIdentity();
	pipeline.ortho(-1.0, 1.0, -1.0, 1.0, 1, 100);
	convTopShades = new shader("../v_shader/convolutionShaderTop.vs","../f_shader/convolutionShaderTop.frag");
	convShades = new shader("../v_shader/convolutionShader.vs","../f_shader/convolutionShader.frag");
	poolingTopShades = new shader("../v_shader/poolingShaderTop.vs","../f_shader/poolingShaderTop.frag");
	poolingShades = new shader("../v_shader/poolingShader.vs","../f_shader/poolingShader.frag");
	resizeTopShades = new shader("../v_shader/resizeTopShader.vs","../f_shader/resizeTopShader.frag");
	displayShades = new shader("../v_shader/displayShader.vs","../f_shader/displayShader.frag");
	poolToConvTopShades = new shader("../v_shader/poolToConvTopShader.vs","../f_shader/poolToConvTopShader.frag");
	convBackTopShades = new shader("../v_shader/convBackTopShader.vs","../f_shader/convBackTopShader.frag");
	poolToConvShades = new shader("../v_shader/poolToConvShader.vs","../f_shader/poolToConvShader.frag");
	resizeShades = new shader("../v_shader/resizeShader.vs","../f_shader/resizeShader.frag");	
	convBackShades = new shader("../v_shader/convBackShader.vs","../f_shader/convBackShader.frag");


	scene = new meshLoader();
	digitSet = new idxFileReader("../models/train-images.idx3-ubyte", "../models/train-labels.idx1-ubyte");

	colorImage = createTexture(SCREEN_WIDTH, SCREEN_HEIGHT);

	convImageA = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	convImageB = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	poolingImageA = createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	poolingImageB = createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	convTopImage = createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	poolingTopImage = createTexture(digitSet->getImageWidth()/4.0, digitSet->getImageHeight()/4.0, true);

	gradientTexture = createRGBTexture(digitSet->getImageWidth()/4.0, digitSet->getImageHeight()/4.0);

	resizeTopPoolImage = createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	resizeTopGradientImage = createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	poolToConvTopImage =  createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	convBackTopImageA = createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	convBackTopImageB = createTexture(digitSet->getImageWidth()/2.0, digitSet->getImageHeight()/2.0, true);
	
	resizePoolImageA = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	resizePoolImageB = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	resizeGradientImageA = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	resizeGradientImageB = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	
	poolToConvImageA = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	poolToConvImageB = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);

	convBackImage = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), true);
	
	//for fully connected layer
	int inputLayerSize = 3 * (digitSet->getImageWidth()/4) * (digitSet->getImageHeight()/4);
	int hiddenLayerCount = 3;
	int outputLayerCount = 10;
	int numCellsPerLayer = 80;
	mlpObject = new mlp(inputLayerSize, hiddenLayerCount, outputLayerCount, numCellsPerLayer);

	convPixelBuffer = mlpObject->getPtrToInput();	//tex_w * tex_h * 4, tex_w = width()/4
	labelBuffer = mlpObject->getPtrToLabel();
	convGradientBuffer = mlpObject->getPtrToInputGradient();		

	//defining frameBuffers with attached textures as render targets
	glGenFramebuffers(1, &colorFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,colorFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,colorImage,0);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	glGenFramebuffers(1, &convFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,convFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,convImageA,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,convImageB,0);	
	GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, bufs);
	
	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);

	glGenFramebuffers(1, &poolingFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,poolingFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,poolingImageA,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,poolingImageB,0);
	GLenum buf[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, buf);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);

	glGenFramebuffers(1, &convTopFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,convTopFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,convTopImage,0);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);

	glGenFramebuffers(1, &poolingTopFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,poolingTopFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,poolingTopImage,0);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);

	glGenFramebuffers(1, &resizeTopFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER, resizeTopFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, resizeTopPoolImage,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D, resizeTopGradientImage,0);
	GLenum bufTopMap[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, bufTopMap);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);
	
	glGenFramebuffers(1, &poolToConvTopFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,poolToConvTopFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,poolToConvTopImage,0);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);

	glGenFramebuffers(1, &convBackTopFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,convBackTopFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,convBackTopImageA,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,convBackTopImageB,0);
	GLenum bufConvTop[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, bufConvTop);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	glGenFramebuffers(1, &resizeFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER, resizeFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, resizePoolImageA,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D, resizePoolImageB,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D, resizeGradientImageA,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT3,GL_TEXTURE_2D, resizeGradientImageB,0);
	GLenum bufMap[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
	glDrawBuffers(4, bufMap);


	glGenFramebuffers(1, &poolToConvFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,poolToConvFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,poolToConvImageA,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,poolToConvImageB,0);
	GLenum poolConvBuf[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, poolConvBuf);

	glGenFramebuffers(1, &convBackFBO);
	glEnable(GL_TEXTURE_2D);
	glBindFramebuffer(GL_FRAMEBUFFER,convBackFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,convBackImage,0);

	fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(fbStatus != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer is not OK, status=" << fbStatus << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER,0);



	{
		std::vector<unsigned int> indices;
		std::vector<vertexData> vertices;
		vertexData tmp;
		//1.
		tmp.position.change(-1.0,1.0,-1.0);
		tmp.U=0;
		tmp.V=0;
		vertices.push_back(tmp);
		//2.
		tmp.position.change(-1.0,-1.0,-1.0);
		tmp.U=0;
		tmp.V=1;
		vertices.push_back(tmp);
		//3.
		tmp.position.change(1.0,-1.0,-1.0);
		tmp.U=1;
		tmp.V=1;
		vertices.push_back(tmp);
		//4.
		tmp.position.change(1.0,1.0,-1.0);
		tmp.U=1;
		tmp.V=0;
		vertices.push_back(tmp);
		
		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);		
		
		indices.push_back(0);
		indices.push_back(2);
		indices.push_back(3);
		quadInverted=new mesh(&vertices,&indices);
	}

	{
		std::vector<unsigned int> indices;
		std::vector<vertexData> vertices;
		vertexData tmp;
		//1.
		tmp.position.change(-1.0,1.0,-1.0);
		tmp.U=0;
		tmp.V=1;
		vertices.push_back(tmp);
		//2.
		tmp.position.change(-1.0,-1.0,-1.0);
		tmp.U=0;
		tmp.V=0;
		vertices.push_back(tmp);
		//3.
		tmp.position.change(1.0,-1.0,-1.0);
		tmp.U=1;
		tmp.V=0;
		vertices.push_back(tmp);
		//4.
		tmp.position.change(1.0,1.0,-1.0);
		tmp.U=1;
		tmp.V=1;
		vertices.push_back(tmp);
		
		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);		
		
		indices.push_back(0);
		indices.push_back(2);
		indices.push_back(3);
		quad = new mesh(&vertices,&indices);
	}


	//random initialize kernels
	for(int itr_k = 0; itr_k < 18; itr_k++)
		kernelMatrix_1[itr_k] = glm::mat3(0.15);
	for(int itr_k = 0; itr_k < 18; itr_k++)
		kernelMatrix_2[itr_k] = glm::mat3(0.05);


/*
//gaussian blur
	kernelMatrix[0][0]=1.0/16.0;	kernelMatrix[0][1]=1.0/8.0;	kernelMatrix[0][2]=1.0/16.0;
	kernelMatrix[1][0]=1.0/8.0;	kernelMatrix[1][1]=1.0/4.0;	kernelMatrix[1][2]=1.0/8.0;
	kernelMatrix[2][0]=1.0/16.0;	kernelMatrix[2][1]=1.0/8.0;	kernelMatrix[2][2]=1.0/16.0;

//sharpen
	kernelMatrix[0][0]=0.0;	kernelMatrix[0][1]=-1.0;	kernelMatrix[0][2]=0.0;
	kernelMatrix[1][0]=-1.0;	kernelMatrix[1][1]=5.00;	kernelMatrix[1][2]=-1.0;
	kernelMatrix[2][0]=0.0;	kernelMatrix[2][1]=-1.0;	kernelMatrix[2][2]=0.0;

//edge detection
	kernelMatrix[0][0]=-1.0;	kernelMatrix[0][1]=-1.0;	kernelMatrix[0][2]=-1.0;
	kernelMatrix[1][0]=-1.0;	kernelMatrix[1][1]=8.00;	kernelMatrix[1][2]=-1.0;
	kernelMatrix[2][0]=-1.0;	kernelMatrix[2][1]=-1.0;	kernelMatrix[2][2]=-1.0;
*/
//identity

	kernelMatrix[0][0]=0.0;	kernelMatrix[0][1]=0.0;	kernelMatrix[0][2]=0.0;
	kernelMatrix[1][0]=0.0;	kernelMatrix[1][1]=1.0;	kernelMatrix[1][2]=0.0;
	kernelMatrix[2][0]=0.0;	kernelMatrix[2][1]=0.0;	kernelMatrix[2][2]=0.0;



	glClearColor(0.25, 0.25, 0.25, 1);
	pipeline.matrixMode(MODEL_MATRIX);
	glBindFramebuffer(GL_FRAMEBUFFER,colorFBO);
		displayShades->useShader();
		glClear(GL_COLOR_BUFFER_BIT);
		pipeline.updateMatrices(displayShades->getProgramId());
		scene->draw(displayShades->getProgramId());
		glBindTexture(GL_TEXTURE_2D, 0);
		displayShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	 imgCount = digitSet->getImageCount();
	glClearColor(0.0, 0.0, 0.0, 1.0);
	
	 digitImage = digitSet->getImage(imageID);
	 label = digitSet->getImageLabel(imageID);
	digitTexture = createTexture(digitSet->getImageWidth(), digitSet->getImageHeight(), digitImage);
	// prevImageID = imageID;

}




std::clock_t displayStart, displayEnd;
void display()
{
	if(imageID != prevImageID && imageID < imgCount)
	{
	 displayStart = std::clock();

	 digitImage = digitSet->getImage(imageID);
	 label = digitSet->getImageLabel(imageID);
	 std::cout<<std::endl<<" label "<<label<<std::endl;
	 updateTextureContent(digitTexture, digitSet->getImageWidth(), digitSet->getImageHeight(), digitImage);
	 prevImageID = imageID;
		
	labelBuffer[label] = 1.0;

	glBindFramebuffer(GL_FRAMEBUFFER,convFBO);
		convShades->useShader();
		glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, digitTexture);
		glUniform1i(glGetUniformLocation(convShades->getProgramId(),"grayInputImage"),0);
		
		glUniform1f(glGetUniformLocation(convShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth());
		glUniform1f(glGetUniformLocation(convShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight());
		
		glUniform1f(glGetUniformLocation(convShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(convShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glUniformMatrix3fv(glGetUniformLocation(convShades->getProgramId(),"kernelMatrix"), 18, GL_FALSE, &kernelMatrix_1[0][0][0]);

		glBindFragDataLocation(convShades->getProgramId(), 0, "texA");
		glBindFragDataLocation(convShades->getProgramId(), 1, "texB");

		pipeline.updateMatrices(convShades->getProgramId());
		quad->draw(convShades->getProgramId());
		glBindTexture(GL_TEXTURE_2D, 0);		
		convShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);



	glBindFramebuffer(GL_FRAMEBUFFER,poolingFBO);
		poolingShades->useShader();
		glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, convImageA);
		glUniform1i(glGetUniformLocation(poolingShades->getProgramId(),"texA"),0);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, convImageB);
		glUniform1i(glGetUniformLocation(poolingShades->getProgramId(),"texB"),1);
		
		glUniform1f(glGetUniformLocation(poolingShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth());
		glUniform1f(glGetUniformLocation(poolingShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight());
		
		glUniform1f(glGetUniformLocation(poolingShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(poolingShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glBindFragDataLocation(poolingShades->getProgramId(), 0, "poolTexA");
		glBindFragDataLocation(poolingShades->getProgramId(), 1, "poolTexB");

		pipeline.updateMatrices(poolingShades->getProgramId());
		quad->draw(poolingShades->getProgramId());
		glBindTexture(GL_TEXTURE_2D, 0);
		poolingShades->delShader();
	 glBindFramebuffer(GL_FRAMEBUFFER,0);


	glBindFramebuffer(GL_FRAMEBUFFER,convTopFBO);
		convTopShades->useShader();
		glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, poolingImageA);
		glUniform1i(glGetUniformLocation(convTopShades->getProgramId(),"firstInputImage"),0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, poolingImageB);
		glUniform1i(glGetUniformLocation(convTopShades->getProgramId(),"secondInputImage"),1);

		//inputTextureSize 60% sure
		glUniform1f(glGetUniformLocation(convTopShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth()/2.0);
		glUniform1f(glGetUniformLocation(convTopShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight()/2.0);
		
		glUniform1f(glGetUniformLocation(convTopShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(convTopShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glUniformMatrix3fv(glGetUniformLocation(convTopShades->getProgramId(),"kernelMatrix"), 18, GL_FALSE, &kernelMatrix_2[0][0][0]);

		glBindFragDataLocation(convTopShades->getProgramId(), 0, "texA");

		pipeline.updateMatrices(convTopShades->getProgramId());
		quad->draw(convTopShades->getProgramId());
		glBindTexture(GL_TEXTURE_2D, 0);		
		convTopShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);
	
	glBindFramebuffer(GL_FRAMEBUFFER,poolingTopFBO);
	poolingTopShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, convTopImage);
		glUniform1i(glGetUniformLocation(poolingTopShades->getProgramId(),"texA"),0);

		glUniform1f(glGetUniformLocation(poolingTopShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth()/2.0);
		glUniform1f(glGetUniformLocation(poolingTopShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight()/2.0);
		
		glUniform1f(glGetUniformLocation(poolingTopShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(poolingTopShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glBindFragDataLocation(poolingTopShades->getProgramId(), 0, "poolTex");
		glBindFragDataLocation(poolingTopShades->getProgramId(), 1, "poolMap");

	pipeline.updateMatrices(poolingTopShades->getProgramId());
	quad->draw(poolingTopShades->getProgramId());
	poolingTopShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	//extract 7X7 pixel values from poolingTopImage Texture
	glBindFramebuffer(GL_FRAMEBUFFER, poolingTopFBO);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, digitSet->getImageWidth()/4, digitSet->getImageHeight()/4, GL_RGB, GL_FLOAT, convPixelBuffer);		//123
	glReadBuffer(GL_BACK);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);



	mlpObject->learnFromMLP(0.85, label, imageID);
	 labelBuffer[label] = 0.0;
	
/*	std::cout<<std::endl<<" after "<<std::endl;

	for(int it = 0; it < digitSet->getImageWidth()/4 * digitSet->getImageHeight()/4; it++)
	{
		std::cout<<std::endl;
		std::cout<<" "<<it<<" "<<convGradientBuffer[3*it+0]<<"  "<<convGradientBuffer[3*it+1]<<"  "<<convGradientBuffer[3*it+2];
	}
*/
	updateTextureContent(gradientTexture, digitSet->getImageWidth()/4, digitSet->getImageHeight()/4, convGradientBuffer);

	
	glBindFramebuffer(GL_FRAMEBUFFER, resizeTopFBO);
	resizeTopShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, poolingTopImage);
		glUniform1i(glGetUniformLocation(resizeTopShades->getProgramId(),"poolImage"),0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, gradientTexture);
		glUniform1i(glGetUniformLocation(resizeTopShades->getProgramId(),"gradImage"),1);

		glUniform1f(glGetUniformLocation(resizeTopShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth()/2.0);
		glUniform1f(glGetUniformLocation(resizeTopShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight()/2.0);
		
		glUniform1f(glGetUniformLocation(resizeTopShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(resizeTopShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glBindFragDataLocation(resizeTopShades->getProgramId(), 0, "poolTex");
		glBindFragDataLocation(resizeTopShades->getProgramId(), 1, "gradTex");

	pipeline.updateMatrices(resizeTopShades->getProgramId());
	quad->draw(resizeTopShades->getProgramId());
	resizeTopShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);

/*
	glBindFramebuffer(GL_FRAMEBUFFER, resizeTopFBO);
	glReadBuffer(GL_COLOR_ATTACHMENT1);
	glReadPixels(0, 0, digitSet->getImageWidth()/2, digitSet->getImageHeight()/2, GL_RGBA, GL_FLOAT, testBuffer);		//123
	glReadBuffer(GL_BACK);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	std::cout<<std::endl<<" after "<<std::endl;

	for(int it = 0; it < digitSet->getImageWidth()/2 * digitSet->getImageHeight()/2; it++)
	{
		std::cout<<std::endl;
		std::cout<<" "<<it<<" "<<testBuffer[4*it+0]<<"  "<<testBuffer[4*it+1]<<"  "<<testBuffer[4*it+2]<<"  "<<testBuffer[4*it+3];
	}
*/

	
	glBindFramebuffer(GL_FRAMEBUFFER, poolToConvTopFBO);
	poolToConvTopShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resizeTopPoolImage);
		glUniform1i(glGetUniformLocation(poolToConvTopShades->getProgramId(),"poolImage"),0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, resizeTopGradientImage);
		glUniform1i(glGetUniformLocation(poolToConvTopShades->getProgramId(),"gradImage"),1);
		
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, convTopImage);
		glUniform1i(glGetUniformLocation(poolToConvTopShades->getProgramId(),"convImage"),2);

		glUniform1f(glGetUniformLocation(poolToConvTopShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth()/2.0);
		glUniform1f(glGetUniformLocation(poolToConvTopShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight()/2.0);
		
		glUniform1f(glGetUniformLocation(poolToConvTopShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(poolToConvTopShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glBindFragDataLocation(poolToConvTopShades->getProgramId(), 0, "gradTex");

	pipeline.updateMatrices(poolToConvTopShades->getProgramId());
	quad->draw(poolToConvTopShades->getProgramId());
	poolToConvTopShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	glBindFramebuffer(GL_FRAMEBUFFER, convBackTopFBO);
	convBackTopShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, poolToConvTopImage);
		glUniform1i(glGetUniformLocation(convBackTopShades->getProgramId(),"inputImage"),0);

		glUniform1f(glGetUniformLocation(convBackTopShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth()/2.0);
		glUniform1f(glGetUniformLocation(convBackTopShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight()/2.0);
		
		glUniform1f(glGetUniformLocation(convBackTopShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(convBackTopShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);
	glUniformMatrix3fv(glGetUniformLocation(convBackTopShades->getProgramId(),"kernelMatrix"), 18, GL_FALSE, &kernelMatrix_2[0][0][0]);

		glBindFragDataLocation(convBackTopShades->getProgramId(), 0, "convBackTexA");
		glBindFragDataLocation(convBackTopShades->getProgramId(), 1, "convBackTexB");

	pipeline.updateMatrices(convBackTopShades->getProgramId());
	quad->draw(convBackTopShades->getProgramId());
	convBackTopShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	//resize gradient image and pooling image to match texWidth and texHeight
	glBindFramebuffer(GL_FRAMEBUFFER, resizeFBO);
	resizeShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, poolingImageA);
		glUniform1i(glGetUniformLocation(resizeShades->getProgramId(),"poolImageA"),0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, poolingImageB);
		glUniform1i(glGetUniformLocation(resizeShades->getProgramId(),"poolImageB"),1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, convBackTopImageA);
		glUniform1i(glGetUniformLocation(resizeShades->getProgramId(),"gradImageA"),2);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, convBackTopImageB);
		glUniform1i(glGetUniformLocation(resizeShades->getProgramId(),"gradImageB"),3);


		glUniform1f(glGetUniformLocation(resizeShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth());
		glUniform1f(glGetUniformLocation(resizeShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight());
		
		glUniform1f(glGetUniformLocation(resizeShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(resizeShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glBindFragDataLocation(resizeShades->getProgramId(), 0, "poolTexA");
		glBindFragDataLocation(resizeShades->getProgramId(), 1, "poolTexB");
		glBindFragDataLocation(resizeShades->getProgramId(), 2, "gradTexA");
		glBindFragDataLocation(resizeShades->getProgramId(), 3, "gradTexB");

	pipeline.updateMatrices(resizeShades->getProgramId());
	quad->draw(resizeShades->getProgramId());
	resizeShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	glBindFramebuffer(GL_FRAMEBUFFER, poolToConvFBO);
	poolToConvShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resizePoolImageA);
		glUniform1i(glGetUniformLocation(poolToConvShades->getProgramId(),"poolImageA"),0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, resizePoolImageB);
		glUniform1i(glGetUniformLocation(poolToConvShades->getProgramId(),"poolImageB"),1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, resizeGradientImageA);
		glUniform1i(glGetUniformLocation(poolToConvShades->getProgramId(),"gradImageA"),2);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, resizeGradientImageB);
		glUniform1i(glGetUniformLocation(poolToConvShades->getProgramId(),"gradImageB"),3);

		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, convImageA);
		glUniform1i(glGetUniformLocation(poolToConvShades->getProgramId(),"convImageA"),4);

		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, convImageB);
		glUniform1i(glGetUniformLocation(poolToConvShades->getProgramId(),"convImageB"),5);

		glUniform1f(glGetUniformLocation(poolToConvShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth());
		glUniform1f(glGetUniformLocation(poolToConvShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight());
		
		glUniform1f(glGetUniformLocation(poolToConvShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(poolToConvShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);

		glBindFragDataLocation(poolToConvShades->getProgramId(), 0, "gradientTexA");
		glBindFragDataLocation(poolToConvShades->getProgramId(), 1, "gradientTexB");

	pipeline.updateMatrices(poolToConvShades->getProgramId());
	quad->draw(poolToConvShades->getProgramId());
	poolToConvShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);


	glBindFramebuffer(GL_FRAMEBUFFER, convBackFBO);
	convBackShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, poolToConvImageA);
		glUniform1i(glGetUniformLocation(convBackShades->getProgramId(),"firstInputImage"),0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, poolToConvImageB);
		glUniform1i(glGetUniformLocation(convBackShades->getProgramId(),"secondInputImage"),1);

		glUniform1f(glGetUniformLocation(convBackShades->getProgramId(),"TEX_WIDTH"), digitSet->getImageWidth());
		glUniform1f(glGetUniformLocation(convBackShades->getProgramId(),"TEX_HEIGHT"), digitSet->getImageHeight());
		
		glUniform1f(glGetUniformLocation(convBackShades->getProgramId(),"SCREEN_WIDTH"), SCREEN_WIDTH);
		glUniform1f(glGetUniformLocation(convBackShades->getProgramId(),"SCREEN_HEIGHT"), SCREEN_HEIGHT);
	glUniformMatrix3fv(glGetUniformLocation(convBackShades->getProgramId(),"kernelMatrix"), 18, GL_FALSE, &kernelMatrix_1[0][0][0]);

		glBindFragDataLocation(convBackShades->getProgramId(), 0, "convBackTex");

	pipeline.updateMatrices(convBackShades->getProgramId());
	quad->draw(convBackShades->getProgramId());
	convBackShades->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER,0);

	displayEnd = std::clock();
	std::cout<<std::endl<<" time taken "<<( displayEnd - displayStart ) / (double) CLOCKS_PER_SEC<<std::endl;



	}


	//display ouput, less changes on produced results
	glClearColor(1, 0, 0, 1);
	displayShades->useShader();
	glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, convBackImage);
		glUniform1i(glGetUniformLocation(displayShades->getProgramId(),"texture0"),0);
	pipeline.updateMatrices(displayShades->getProgramId());
	quadInverted->draw(displayShades->getProgramId());
	displayShades->delShader();
	
}


int main()
{

	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

	gWindow = SDL_CreateWindow("SDL_COLLIDE", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
	SDL_GLContext gContext = SDL_GL_CreateContext(gWindow);
	glewExperimental = GL_TRUE;
	glewInit();
	SDL_GL_SetSwapInterval( 1 );
	gScreenSurface = SDL_GetWindowSurface( gWindow );

	bool running=true;
	SDL_Event event;	
	init();

	while(running)
	{
		while(SDL_PollEvent(&event))
		{
			switch(event.type)
			{
				case SDL_QUIT:
				running = false;
				break;
	
				case SDL_KEYDOWN:
				switch(event.key.keysym.sym)
					{
						case SDLK_ESCAPE:
							running=false;
							break;
						case SDLK_RIGHT:	
							imageID++;
							break;
						case SDLK_LEFT:
							imageID--;
							break;

						
					}
	
			}
		}

		display();
		SDL_GL_SwapWindow(gWindow);

	}

	delete mlpObject;
	delete resizeTopShades;
	delete resizeShades;
	delete convShades;
	delete convBackShades;
	delete convTopShades;
	delete poolingShades;
	delete poolingTopShades;
	delete poolToConvTopShades;
	delete poolToConvShades;
	delete convBackTopShades;
	delete displayShades;
	delete scene;
	delete quad, quadInverted;
	delete digitSet;
	SDL_FreeSurface(gScreenSurface);
	SDL_GL_DeleteContext(gContext);
	SDL_DestroyWindow(gWindow);
	SDL_Quit();

	return 0;
}

unsigned int createTexture(int w,int h, bool isFloatTex)
{
	unsigned int textureId;
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1,&textureId);
	glBindTexture(GL_TEXTURE_2D,textureId);
	glTexImage2D(GL_TEXTURE_2D,0, (isFloatTex ? GL_RGBA32F : GL_RGBA8), w, h, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER);
	
	int i;
	i=glGetError();
	if(i!=0)
		std::cout << "Error happened while loading the texture: " << gluErrorString(i) << std::endl;
	glBindTexture(GL_TEXTURE_2D,0);
	return textureId;
}

unsigned int createTexture(int w, int h, unsigned char *pixels)
{
	unsigned int textureId;
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1,&textureId);
	glBindTexture(GL_TEXTURE_2D,textureId);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RED,w,h,0,GL_RED,GL_UNSIGNED_BYTE,pixels);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	
	//int i;
	//i=glGetError();
	//if(i!=0)
	//	std::cout << "Error happened while loading the texture: " << gluErrorString(i) << std::endl;
	glBindTexture(GL_TEXTURE_2D,0);
	return textureId;

}

unsigned int createRGBTexture(int w, int h)
{
	unsigned int textureId;
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1,&textureId);
	glBindTexture(GL_TEXTURE_2D,textureId);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB32F,w,h,0,GL_RGB,GL_FLOAT,0);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	
	//int i;
	//i=glGetError();
	//if(i!=0)
	//	std::cout << "Error happened while loading the texture: " << gluErrorString(i) << std::endl;
	glBindTexture(GL_TEXTURE_2D,0);
	return textureId;

}

void updateTextureContent(unsigned int textureId, int w, int h, float* pixels)
{
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,textureId);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB32F,w,h,0,GL_RGB,GL_FLOAT,pixels);
//	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D,0);

}


void updateTextureContent(unsigned int textureId, int w, int h, unsigned char* pixels)
{
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,textureId);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RED,w,h,0,GL_RED,GL_UNSIGNED_BYTE,pixels);
//	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D,0);

}


