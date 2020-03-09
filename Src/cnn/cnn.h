/**
*	Author: Hou Wang
*   PID: A53241783
*	This file defines the data structure for grid
*/

#pragma once
#include <vector>
#include <iostream>
#include <unordered_map>
#define CATEGORIES 15
#define BATCH_SIZE 100
#define EPOCHS 50
namespace cnn {
	class Layer {
		// layer need to specify its own hyperparameters
		// child can specify Activation acFunction;
		// linear layer / FC layer: W,b, activation
		// conv layer: W,b, stride, pad
		// pool layer: f, stride, pad
		// From last layer, we can get A_last, which is used to compute cost
		// each layer need to store 
	public:
		float learningRate;
		float* A; // result of previous activation
		// Activation Shapes: W x H x C
		int curShape[3];
		int prevShape[3];
		Layer(int prev[3], float lr = 0.0075) {
			prevShape[0] = prev[0];
			prevShape[1] = prev[1];
			prevShape[2] = prev[2];
			learningRate = lr;
		}

		virtual void init() = 0;
		// Input: output from previous layer, typically a 3 dim matrix
		// Output: activation/conv/pool result from current layer, typically a 3 dim matrix
		/*virtual std::vector<narray::Narray<float>*> forward(
		std::vector<narray::Narray<float>*> A_prev
		) = 0;*/

		virtual float* forward(float* A_prev) = 0;

		// Input gradient of current layer output
		// for linear layer, the input will be actually dZ
		// for conv layer: it will store A_prev from forward step in order to compute backward
		/*virtual std::vector<narray::Narray<float>*> backward(
		std::vector<narray::Narray<float>*> dA,
		) = 0;*/

		virtual float* backward(float* dA) = 0;
	};

	class ReLU : public Layer {
	private:
		float* W; // in GPU mem
		float* b; // in GPU mem
	public:
		ReLU(int curNeurons, int prev[3], float lr = 0.0075) : Layer(prev, lr) {
			W = nullptr;
			b = nullptr;
			A = nullptr;
			curShape[0] = curNeurons;
			curShape[1] = 1;
			curShape[2] = 1;
			init();
			std::cout << "Relu layer init finished" << std::endl;
		}
		void init();
		//TODO: Implement both:
		float* forward(float* A_prev);
		float* backward(float* dA);
	};

	class SoftMax : public Layer {
	private:
		float* W;
		float* b;
	public:
		SoftMax(int curNeurons, int prev[3], float lr = 0.0075) : Layer(prev, lr) {
			W = nullptr;
			b = nullptr;
			A = nullptr;
			curShape[0] = curNeurons;
			curShape[1] = 1;
			curShape[2] = 1;
			init();
			std::cout << "softmax layer init finished" << std::endl;
		}
		void init();
		//TODO: Implement both:
		float* forward(float* A_prev);
		// update paramters during backward
		float* backward(float* dA);
	};

	class CNN {
		std::vector<Layer*> layers;
		int inputShape[3];
	public:
		CNN(int input[3]) {
			inputShape[0] = input[0];
			inputShape[1] = input[1];
			inputShape[2] = input[2];
		}
		void addLayer(Layer* layer);

		// TODO:implement this
		// compute average cost over the minibatch samples
		// return cost
		float computeCost(float* A_final, float* Y_one_hot);

		// TODO:implement this
		// compute grad from loss function
		// implement cross-entropy loss
		float* computeLossGrad(float* A_final, float* Y_one_hot);

		int correctPredict(float* A_final, float* Y_one_hot);

		void train(std::vector<float*>& X_train,
			std::vector<int>& Y_train,
			int epochs = EPOCHS,
			int batch_size = BATCH_SIZE);

		void evaluate(std::vector<float*>& X_test,
			std::vector<int>& Y_test);
	};
}

#include <curand_kernel.h>
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
		system("pause");													   \
        exit(1);                                                               \
    }                                                                          \
}
