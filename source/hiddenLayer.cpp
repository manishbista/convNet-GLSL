
#include "../header/hiddenLayer.h"


hiddenLayer::hiddenLayer(unsigned int numCell_prev, unsigned int numCell_cur, float* inputVal, float* outputVal, bool isSigmoid)
{
 weights = new float* [numCell_cur];
 for(int i = 0; i < numCell_cur; i++)
 weights[i] = new float[numCell_prev];

 biasWeights = new float[numCell_cur];

 cellObj = new cell* [numCell_cur];
 for(int i = 0; i < numCell_cur; i++)
	cellObj[i] = new cell(numCell_prev, weights[i], inputVal, &(outputVal[i]), biasWeights[i], isSigmoid);

 hiddenLayer::numCell_prev = numCell_prev;
 hiddenLayer::numCell_cur = numCell_cur;
 hiddenLayer::inputVal = inputVal;
 hiddenLayer::outputVal = outputVal;
 hiddenLayer::isHiddenLayer = isSigmoid;
}

void hiddenLayer::forwardPassLayer()
{
	//forward pass for all cells
	for(int i = 0; i < numCell_cur; i++)
	 cellObj[i]-> forwardPassCell();

	if(!isHiddenLayer) 			//if this layer is hidden Layer, then use softmax function
	{
		float cumulativeSum = 0.0;
		for(int i = 0; i < numCell_cur; i++)
		 cumulativeSum += exp(outputVal[i]);
		for(int i = 0; i < numCell_cur; i++)
		 outputVal[i] = exp(outputVal[i]) / cumulativeSum;		
	}
}

void hiddenLayer::backwardPassLayer(float* errGradient_prev, float* errGradient_cur)
{
 float p;
	for(int j = 0; j < numCell_prev; j++)
	{
		errGradient_prev[j] = 0.0;
		for(int i = 0; i < numCell_cur; i++)
		 errGradient_prev[j] += errGradient_cur[i] * weights[i][j];

		//sigmoid gradient s(1-s)
		errGradient_prev[j] *= inputVal[j] * (1 - inputVal[j]);
		//ReLU gradient
		// p = (inputVal[j] == 0.0 ? 0.0 : 1.0);
		//errGradient_prev[j] *= p;
	}
//use errGradient_cur to update weights
	for(int i = 0; i < numCell_cur; i++)
	{
		for(int j = 0; j < numCell_prev; j++)
		  weights[i][j] += errGradient_cur[i] * inputVal[j];
		biasWeights[i] += errGradient_cur[i];
	}
}


hiddenLayer::~hiddenLayer()
{
 delete biasWeights;
 delete[] weights;
 delete[] cellObj;
}


