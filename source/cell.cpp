
#include "../header/cell.h"

cell::cell(unsigned int numCell_prev, float* weights, float* inputVal, float* outputVal, float biasWeight, bool isSigmoid)
{
	cell::numCell_prev = numCell_prev;
	cell::weights = weights;
	cell::inputVal = inputVal;
	cell::outputVal = outputVal;
	cell::isSigmoid = isSigmoid;
	cell::biasWeight = biasWeight;	

	//initialize weights
	for(int i = 0; i < numCell_prev; i++)
		weights[i] = 0.05;
	cell::biasWeight = 0.05;

}

float cell::sigmoid(float varX)
{
	return 1/(1+exp(-varX));

}

float cell::ReLU(float varX)
{
	float p =  (varX > 0.0) ? varX:0.0;
	return p;
}

void cell::forwardPassCell()
{
	float cumulativeSum = 0.0;
	for(int i = 0; i < numCell_prev; i++)
	 cumulativeSum += weights[i] * inputVal[i];
	//std::cout<<" cSum "<<cumulativeSum<<std::endl;
	cumulativeSum += biasWeight;
	if(isSigmoid) *outputVal = sigmoid(cumulativeSum);
	else if(!isSigmoid) *outputVal = cumulativeSum;

}


