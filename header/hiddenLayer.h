
#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include "cell.h"

class hiddenLayer{
 private:
	cell** cellObj;
	float** weights;
	float* biasWeights;
	unsigned int numCell_prev, numCell_cur;
	float *inputVal, *outputVal;
	bool isHiddenLayer;

 public:
	hiddenLayer(unsigned int, unsigned int, float*, float*, bool isSigmoid = true);
	~hiddenLayer();
	void forwardPassLayer();
	void backwardPassLayer(float*, float*);

};


#endif
