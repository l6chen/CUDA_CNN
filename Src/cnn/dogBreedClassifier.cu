/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include "cnn.h"
#include "preprocessor.h"
//#include "narray.h"
#include <string>
#include <iostream>
#include <fstream>
const int SUCCESS = 0;

// main function
int main(int argc, char* argv) {
	using namespace cnn;
	std::string path = "C:/Users/Morligan/Desktop/Githubprj/CUDA/final project/CUDA/datasets";
	std::ofstream out("C:/Users/Morligan/Desktop/Githubprj/CUDA/final project/CUDA/out.txt");
	std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
	//std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
	std::cout << "Start" << std::endl;
	prep::Preprocessor processor(path);

	// Read image data and labels
	processor.loadDataSets();

	//// get trainingData
	//std::vector<narray::Narray<float>*> X_train = getTrainingData();
	//std::vector<narray::Narray<float>*> X_test = getTestData();
	std::vector<float*> X_train = processor.getTrainingData();
	std::vector<float*> X_test = processor.getTestData();
	std::vector<int> Y_train = processor.getTrainingLabels();
	std::vector<int> Y_test = processor.getTestLabels();

	// build model
	int* inputShape = new int[3];
	std::cout << "W:" << processor.getImgWidth() << " "
		<< "H:" << processor.getImgHeight() << " "
		<< "C:" << processor.getImgChannels() << std::endl;
	inputShape[0] = processor.getImgWidth() * processor.getImgHeight() * processor.getImgChannels();
	inputShape[1] = 1;
	inputShape[2] = 1;
	CNN model(inputShape);

	ReLU relu0(32, inputShape);
	model.addLayer(dynamic_cast<Layer *>(&relu0));

	inputShape = relu0.curShape;
	ReLU relu1(16, inputShape);
	model.addLayer(dynamic_cast<Layer *>(&relu1));

	inputShape = relu1.curShape;
	ReLU relu2(8, inputShape);
	model.addLayer(dynamic_cast<Layer *>(&relu2));

	inputShape = relu2.curShape;
	SoftMax sm0(CATEGORIES, inputShape);
	model.addLayer(dynamic_cast<Layer *>(&sm0));
	
	// fit data
	model.train(X_train, Y_train);

	// evaluate
	model.evaluate(X_test, Y_test);
	
	std::cout << "Evaluate complete" << std::endl;
	system("pause");
	return SUCCESS;
}
