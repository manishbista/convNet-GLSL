
#include "../header/mlp.h"


mlp::mlp(int numIn, int numHid, int numOut, int nCells)			//default constructor for a binomial classifier
{
	numInputNode = numIn;
	numOutputNode = numOut;
	numHiddenLayer = numHid;

	numCell = new unsigned int[numHiddenLayer + 2];
	numCell[0] = numInputNode;
	numCell[numHiddenLayer +1] = numOutputNode;
	for(int i = 1; i < numHiddenLayer +1; i++)
	  numCell[i] = nCells;

	ioValues = new float* [numHiddenLayer +2];
	for(int i = 0; i < numHiddenLayer+2; i++)
	  ioValues[i] = new float[numCell[i]];

	errGradient = new float* [numHiddenLayer +2];
	for(int i = 0; i < numHiddenLayer +2; i++)
	  errGradient[i] = new float[numCell[i]];

	hiddenLayerObj = new hiddenLayer* [numHiddenLayer+1];
	for(int i = 0; i < numHiddenLayer; i++)			//all HiddenLayers and +1 for output layer as well
	  hiddenLayerObj[i] = new hiddenLayer(numCell[i], numCell[i+1], ioValues[i], ioValues[i+1]);
	  hiddenLayerObj[numHiddenLayer] = new hiddenLayer(numCell[numHiddenLayer], numCell[numHiddenLayer+1], ioValues[numHiddenLayer], ioValues[numHiddenLayer+1], false);
	
	labelPtr = new float[numOutputNode];	
	for(int i = 0; i < numOutputNode; i++)	labelPtr[i] = 0.0;
}


bool mlp::learnFromMLP(float thresholdValue, int label, int imageID)
{
  unsigned int iterationCount = 0;
  float evalAccuracy = 0.0;			//assume zero accuracy initially
	
	for(int i = 0; i < numHiddenLayer+1; i++){
		hiddenLayerObj[i]->forwardPassLayer(); }

	evalAccuracy = 1.0;
	for(int i = 0; i < numOutputNode; i++)
	{
		errGradient[numHiddenLayer+1][i] = (labelPtr[i] - ioValues[numHiddenLayer+1][i]);
		evalAccuracy *= pow(ioValues[numHiddenLayer+1][i], labelPtr[i]);
	}
	//evalAccuracy is the maximum value of probability


	for(int i = numHiddenLayer; i >= 0; i--)
		hiddenLayerObj[i]->backwardPassLayer(errGradient[i], errGradient[i+1]);
	 std::cout<<std::endl<<" ID "<<imageID<<" eval Accuracy "<<evalAccuracy<<std::endl;
/*
	std::cout<<" after forward pass "<<std::endl;
	for(int i = 0; i < numHiddenLayer+2; i++)
	{
		for(int j = 0; j < numCell[i]; j++)
		{
			std::cout<<" for cell i = "<<i<<" j = "<<j<<"  val is "<<ioValues[i][j]<<std::endl;		

		}

	}



	std::cout<<" after backward pass "<<std::endl;
	for(int i = 0; i < numHiddenLayer + 2; i++)
	{
		for(int j = 0; j < numCell[i]; j++)
		{
			std::cout<<" for cell i = "<<i<<" j = "<<j<<"  val is "<<errGradient[i][j]<<std::endl;		

		}

	}
*/

	if(evalAccuracy < thresholdValue) return true;
	else return false;

}

void mlp::testMLP()
{
	std::cout<<std::endl<<" testing procedure "<<std::endl;
	for(int i = 0; i < numHiddenLayer+1; i++)
	hiddenLayerObj[i]->forwardPassLayer();

	//display output
	std::cout<<" after forward pass "<<std::endl;
	for(int i = 0; i < numHiddenLayer+2; i++)
	{
		for(int j = 0; j < numCell[i]; j++)
		{
			std::cout<<" for cell i = "<<i<<" j = "<<j<<"  val is "<<ioValues[i][j]<<std::endl;		

		}

	}

}

mlp::~mlp()
{

	delete labelPtr;
	delete numCell;
	delete[] ioValues;
	delete[] errGradient;
	delete[] hiddenLayerObj;
}
