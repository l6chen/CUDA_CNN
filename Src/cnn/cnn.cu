/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include "cnn.h"
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <cmath>
#include <ctime>

#define CURSTATE_LEN 233
#define BLOCK_SIZE 16
namespace cnn {
	const std::string DELIMITER = "=====================================================";
	__global__ void matrixMultiplication(float *a, float *b, float *c, int m, int n, int k)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		float sum = 0;
		if (col < k && row < m)
		{
			for (int i = 0; i < n; i++)
			{
				sum += a[row * n + i] * b[i * k + col];
			}
			c[row * k + col] = sum;
		}
	}

	// A: m x n, B:m x 1
	__global__ void matrixBias(float *d_A, float *d_B, int m, int n) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int idy = threadIdx.y + blockIdx.y * blockDim.y;
		if (idx < n && idy < m) {
			int index = idy * n + idx;
			d_A[index] += d_B[idy];
		}
	}

	__global__ void reluActivation(float *d_AL, int m) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < m) {
			float val = d_AL[idx];
			d_AL[idx] = val > 0 ? val : 0;
		}
	}

	__global__ void softmaxActivation(float *d_AL, int m) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		float exp = expf(d_AL[idx]);
		d_AL[idx] = exp;
		float sum = 0;

		// compute sum
		int startIdx = idx % CATEGORIES;
		int end = startIdx + CATEGORIES;
		for (int i = startIdx; i < end; ++i) {
			sum += d_AL[i];
		}

		float sm = exp / sum;
		d_AL[idx] = sm;
	}

	__global__ void mulTranspose(float *odata, float *idata, int width, int height)
	{
		__shared__ float block[BLOCK_SIZE][BLOCK_SIZE + 1];

		// read the matrix tile into shared memory
		unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
		if ((xIndex < width) && (yIndex < height))
		{
			unsigned int index_in = yIndex * width + xIndex;
			block[threadIdx.y][threadIdx.x] = idata[index_in];
		}

		__syncthreads();

		// write the transposed matrix tile to global memory
		xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
		yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
		if ((xIndex < height) && (yIndex < width))
		{
			unsigned int index_out = yIndex * height + xIndex;
			odata[index_out] = block[threadIdx.x][threadIdx.y];
		}
	}

	__global__ void setup_rand(curandState* state, int w, int h, int c) {
		int idxX = blockIdx.x * blockDim.x + threadIdx.x;
		int idxY = blockIdx.x * blockDim.y + threadIdx.y;
		int idxZ = blockIdx.z * blockDim.z + threadIdx.z;
		if (idxX < w && idxY < h && idxZ < c) {
			int idx = idxZ * (w * h) + idxY * w + idxX;
			// TODO: in final run, remove CURSTATE_LEN limit
			int curS = idx;
			curand_init((unsigned long long)clock() + curS, curS, 0, &state[curS]);
		}
	}

	__global__ void random_init(curandState* state, float* W, int w, int h, int c, float range) {
		int idxX = blockIdx.x * blockDim.x + threadIdx.x;
		int idxY = blockIdx.y * blockDim.y + threadIdx.y;
		int idxZ = blockIdx.z * blockDim.z + threadIdx.z;
		if (idxX < w && idxY < h && idxZ < c) {
			int idx = idxZ * (w * h) + idxY * w + idxX;
			curandState localState = state[idx];
			float val = curand_uniform(&localState) * range;
			W[idx] = val;
			state[idx] = localState;
		}
	}
	// A = Y: m x n
	__global__ void calCrossEntropyLoss (float* d_out, float *d_A, float *d_Y, int m, int n) {
		int nidx = threadIdx.x + blockDim.x * blockIdx.x;
		if (nidx < n) {
			for (int midx = 0; midx < m; ++midx) {
				int curIdx = midx * n + nidx;
				if (d_Y[curIdx] != 0) {
					d_out[midx] += -logf(d_A[curIdx]);
					break;
				}
			}
		}
	}
	// A = Y: m x n
	__global__ void elementWiseMatrixDeduction(float* d_out, float* d_A, float *d_Y, int m, int n) {
		int mIdx = threadIdx.y + blockDim.y * blockIdx.y;
		int nIdx = threadIdx.x + blockDim.x * blockIdx.x;
		if (mIdx < m && nIdx < n) {
			int idx = mIdx * n + nIdx;
			d_out[idx] = d_A[idx] - d_Y[idx];
		}
	}

	__global__ void matrixSumToOneAxis (float *d_dB, float *d_dZ, int m, int n) {
		int mIdx = threadIdx.x + blockIdx.x * blockDim.x;
		if (mIdx < m) {
			float sum = 0;
			for (int i = 0; i < n; ++i) {
				int idx = mIdx * n + i;
				d_dB[mIdx] += d_dZ[idx];
			}
		}
	}

	__global__ void reluGrad(float *d_dZ, int m, int n) {
		int mIdx = threadIdx.y + blockIdx.y * blockDim.y;
		int nIdx = threadIdx.x + blockIdx.x * blockDim.x;
		if (mIdx < m && nIdx < n) {
			int idx = mIdx * n + nIdx;
			float val = d_dZ[idx];
			d_dZ[idx] = (val > 0) ? val : 0;
		}
	}

	__global__ void elementWiseMatrixMultiplication(float *d_dW, float learningRate, int m, int n) {
		int mIdx = threadIdx.y + blockIdx.y * blockDim.y;
		int nIdx = threadIdx.x + blockIdx.x * blockDim.x;
		if (mIdx < m && nIdx < n) {
			int idx = mIdx * n + nIdx;
			d_dW[idx] *= learningRate;
		}
	}

	__global__ void countCorrectPredict(int* d_count, float* d_A, float* d_Y, int m, int n) {
		int nIdx = threadIdx.x + blockDim.x * blockIdx.x;
		if (nIdx < n) {
			bool isCorrect = false;
			float max = FLT_MIN;
			for (int i = 0; i < m; ++i) {
				int idx = m * i + nIdx;
				float cur = d_A[idx];
				float curLabel = d_Y[idx];
				if (cur > max) {
					isCorrect = d_Y[idx] != 0;
					max = cur;
				}
			}

			if (isCorrect) d_count[nIdx] = 1;
			else d_count[nIdx] = 0;
		}
	}
	// ==========================================================================
	// Kernel function wrappers:
	void transpose(float* A_v, int height, int width) {
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
		float *d_A_v, *d_out;

		CHECK(cudaMalloc((void**)&d_A_v, height * width * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_out, height * width * sizeof(float)));
		CHECK(cudaMemcpy(d_A_v, A_v, height * width * sizeof(float), cudaMemcpyHostToDevice));
		mulTranspose <<<grid, block >>> (d_out, d_A_v, width, height);
		CHECK(cudaMemcpy(A_v, d_out, height * width * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(d_A_v);
		cudaFree(d_out);
	}

	// A: m x n, B: n x k
	void matrixMul(float* out, float* A, float* B, int m, int n, int k) {
		float *d_out, *d_A, *d_B;
		CHECK(cudaMalloc((void**)&d_out, m * k * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_B, n * k * sizeof(float)));

		CHECK(cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

		matrixMultiplication << <grid, block >> > (d_A, d_B, d_out, m, n, k);
		CHECK(cudaMemcpy(out, d_out, m * k * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_out);
		cudaFree(d_A);
		cudaFree(d_B);
	}

	// A: m x n, B: m x 1
	void matrixAddBias(float* A, float* B, int m, int n) {
		float* d_A, *d_B;
		CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_B, m * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, B, m * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
		matrixBias <<<grid, block >>> (d_A, d_B, m, n);
		CHECK(cudaMemcpy(A, d_A, m * n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
		cudaFree(d_B);
	}

	// aply relu activation on all A units
	void relu(float* A, int len) {
		float* d_A;
		CHECK(cudaMalloc((void**)&d_A, len * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A, len * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE * BLOCK_SIZE);
		dim3 grid((len + block.x - 1) / block.x);
		reluActivation <<<grid, block >>> (d_A, len);
		CHECK(cudaMemcpy(A, d_A, len * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
	}

	void softmax(float* A, int len) {
		float *d_A;
		CHECK(cudaMalloc((void**)&d_A, len * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A, len * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(150);
		dim3 grid((len + block.x - 1) / block.x);
		softmaxActivation <<<grid, block >>> (d_A, len);
		CHECK(cudaMemcpy(A, d_A, len * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
	}
	// A = Y: m x n
	float crossEntropyLoss(float *A_final, float *Y_one_hot, int m, int n) {
		float *d_A, *d_Y, *d_out;
		float *out = (float *)malloc(m * sizeof(float));
		CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_Y, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_out, m * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A_final, m *n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Y, Y_one_hot, m * n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);
		calCrossEntropyLoss <<<grid, block >>> (d_out, d_A, d_Y, m, n);
		CHECK(cudaMemcpy(out, d_out, m * sizeof(float), cudaMemcpyDeviceToHost));

		float cost = 0;
		for (int i = 0; i < m; ++i) {
			cost += out[i];
		}
		cost /= m;
		cudaFree(d_A);
		cudaFree(d_Y);
		cudaFree(d_out);
		return cost;
	}
	// A = Y: m x n
	float* elementWiseMinus(float *A_final, float *Y_one_hot, int m, int n) {
		float *d_A, *d_Y, *d_out;
		float *out = (float *)malloc(m * n * sizeof(float));
		CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_Y, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_out, m * n * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A_final, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Y, Y_one_hot, m * n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
		elementWiseMatrixDeduction << <grid, block >> > (d_out, d_A, d_Y, m, n);
		CHECK(cudaMemcpy(out, d_out, m * n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
		cudaFree(d_Y);
		cudaFree(d_out);
		return out;
	}
	// dB: m x 1, dZ: m x n
	void matrixSum(float* dB, float* dZ, int m, int n) {
		float *d_dB, *d_dZ;
		CHECK(cudaMalloc((void**)&d_dZ, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_dB, m * sizeof(float)));
		CHECK(cudaMemcpy(d_dZ, dZ, m * n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE);
		dim3 grid((m + block.x - 1) / block.x);
		matrixSumToOneAxis << <grid, block >> > (d_dB, d_dZ, m, n);
		CHECK(cudaMemcpy(dB, d_dB, m * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_dB);
		cudaFree(d_dZ);
	}

	// dZ: m x n
	void reluGradFilter(float* dZ, int m, int n) {
		float *d_dZ;
		CHECK(cudaMalloc((void**)&d_dZ, m * n * sizeof(float)));
		CHECK(cudaMemcpy(d_dZ, dZ, m * n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((n + block.x - 1)/ block.x, (m + block.y - 1) / block.y);
		reluGrad <<<grid, block >>> (d_dZ, m, n);
		CHECK(cudaMemcpy(dZ, d_dZ, m * n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_dZ);
	}

	void updateW(float* W, float *dW, int m, int n, float learningRate) {
		float *d_W, *d_dW, *d_out;
		CHECK(cudaMalloc((void**)&d_W, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_dW, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_out, m * n * sizeof(float)));
		CHECK(cudaMemcpy(d_W, W, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_dW, dW, m * n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
		elementWiseMatrixMultiplication <<<grid, block >>> (d_dW, (float)learningRate / BATCH_SIZE, m, n);
		elementWiseMatrixDeduction <<<grid, block >>> (d_out, d_W, d_dW, m, n);
		CHECK(cudaMemcpy(W, d_out, m * n * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(d_W);
		cudaFree(d_dW);
		cudaFree(d_out);
	}

	void updateB(float* B, float *dB, int m, float learningRate) {
		float *d_B, *d_dB, *d_out;
		CHECK(cudaMalloc((void**)&d_B, m * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_dB, m * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_out, m * sizeof(float)));
		CHECK(cudaMemcpy(d_B, B, m * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_dB, dB, m * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(1, BLOCK_SIZE);
		dim3 grid(1, (m + block.y - 1) / block.y);
		elementWiseMatrixMultiplication << <grid, block >> > (d_dB, (float)learningRate / BATCH_SIZE, m, 1);
		elementWiseMatrixDeduction << <grid, block >> > (d_out, d_B, d_dB, m, 1);
		CHECK(cudaMemcpy(B, d_out, m * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(d_B);
		cudaFree(d_dB);
		cudaFree(d_out);
	}

	int getCorrectCount(float* A_final, float* Y_one_hot, int m, int n) {
		float* d_A, *d_Y;
		int *count, *d_count;
		count = (int *)malloc(n * sizeof(int));
		CHECK(cudaMalloc((void**)&d_A, m * n *sizeof(float)));
		CHECK(cudaMalloc((void**)&d_Y, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&d_count, n * sizeof(int)));
		CHECK(cudaMemcpy(d_A, A_final, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Y, Y_one_hot, m * n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.y - 1) / block.y);
		countCorrectPredict <<<grid, block >>> (d_count, d_A, d_Y, m, n);
		CHECK(cudaMemcpy(count, d_count, n * sizeof(float), cudaMemcpyDeviceToHost));

		int sum = 0;
		for (int i = 0; i < n; ++i) {
			sum += count[i];
		}
		free(count);
		cudaFree(d_A);
		cudaFree(d_Y);
		cudaFree(d_count);
		return sum;
	}

	// ===================================================================================
	// Class implementations
	void CNN::addLayer(Layer* layer) {
		this->layers.push_back(layer);
	}

	void ReLU::init() {
		// conv filter dimension: curNeuron * prevNeuron
		int w = prevShape[0];
		int h = curShape[0];
		int n = w * h;
		float range = sqrtf((float)2 / n);
		dim3 block(32, 8);
		dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

		curandState *randState;
		// set up cuda random generator
		CHECK(cudaMalloc((void**)&randState, n * sizeof(curandState)));
		//test_setup_rand << <grid, block >> > (randState, w, h, 1);
		setup_rand << <grid, block >> > (randState, w, h, 1);

		this->W = (float*)malloc(n * sizeof(float));
		this->b = (float*)malloc(h * sizeof(float));
		memset(b, 0, h * sizeof(float));
		float* d_W;
		CHECK(cudaMalloc((void**) &d_W, n * sizeof(float)));
		random_init <<< grid, block >>> (randState, d_W, w, h, 1, range);
		CHECK(cudaMemcpy(W, d_W, n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_W);
		cudaFree(randState);
	}

	void SoftMax::init() {
		// conv filter dimension: curNeuron * prevNeuron
		int w = this->prevShape[0];
		int h = this->curShape[0];
		int n = w * h;
		float range = sqrtf((float)1 / n);
		dim3 block(32, 8);
		dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

		curandState* randState;
		// set up cuda random generator
		CHECK(cudaMalloc((void**)&randState, n * sizeof(curandState)));
		setup_rand << <grid, block >> > (randState, w, h, 1);

		this->W = (float*)malloc(n * sizeof(float));
		this->b = (float*)malloc(h * sizeof(float));
		memset(b, 0, h * sizeof(float));
		float* d_W;
		CHECK(cudaMalloc((void**)&d_W, n * sizeof(float)));
		random_init << <grid, block >> > (randState, d_W, w, h, 1, range);
		CHECK(cudaMemcpy(this->W, d_W, n * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(d_W);
		cudaFree(randState);
	}

	void initMiniBatch(std::vector<int>& v, int size) {
		for (int i = 0; i < size; ++i) {
			v.push_back(i);
		}
	}
	
	void getCurrentBatch(std::vector<float*>& A,
		std::vector<int>& Y_batch,
		std::vector<int>& miniBatch,
		std::vector<float*>& X_train,
		std::vector<int>& Y_train,
		int mIdx,
		int numOfMiniBatches,
		int batch_size) {

		int startIdx = mIdx * batch_size;
		for (int i = 0; i < batch_size; ++i) {
			int curIdx = startIdx + i;
			int dataIdx = miniBatch[curIdx];
			A.push_back(X_train[dataIdx]);
			Y_batch.push_back(Y_train[dataIdx]);
		}
	}

	// =========================================
	// Helpers
	int predict(float* A, int Y) {
		return 0;
	}

	void vectorize(std::vector<float*>& A, float* A_v, int A_len, std::vector<int>& Y_batch, float* Y_one_hot, int Y_len) {
		for (int i = 0; i < A.size(); ++i) {
			for (int j = 0; j < A_len; ++j) {
				auto tmp = A[i][j];
				A_v[i * A_len + j] = A[i][j];//array overflow, solved by resize img
			}
		}

		for (int i = 0; i < Y_batch.size(); ++i) {
			for (int j = 0; j < Y_len; ++j) {
				int idx = i * Y_len + j;	
				Y_one_hot[idx] = j == Y_batch[i] ? 1 : 0;
			}
		}
	}

	// return activation of last layer A_final
	float* forwardPropagation(float* X , std::vector<Layer*>& layers) {
		float* A = X;
		for (Layer* layer : layers) {
			// A: batch_size * v
			A = layer->forward(A);
		}

		return A;
	}

	void backwardPropagation(float* dZ, std::vector<Layer*>& layers) {
		// backward prop and update parameters
		float* dAL = dZ;
		for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) {
			dAL = (*layer)->backward(dAL);
		}
	}

	void CNN::train(std::vector<float*>& X_train,
		std::vector<int>& Y_train,
		int epochs,
		int batch_size) {
		
		int m = X_train.size();
		for (int eIdx = 0; eIdx < epochs; ++eIdx) {
			// loop through epochs
			int miniBatchCost = 0;
			std::vector<int> miniBatch;
			initMiniBatch(miniBatch, m);
			std::random_shuffle(miniBatch.begin(), miniBatch.end());
			int numOfMiniBatches = (int) X_train.size() / batch_size;
			float avgMiniBatchCost = 0;

			int correctPred = 0;
			std::clock_t start = std::clock();
			double duration;

			for (int mIdx = 0; mIdx < numOfMiniBatches; ++mIdx) {
				// forward propagate m samples in current batch
				// compute avg cost 
				std::vector<float*> A;
				std::vector<int> Y_batch;
				getCurrentBatch(A, Y_batch, miniBatch, X_train, Y_train, mIdx, numOfMiniBatches, batch_size);
				
				// Y_batch stores current Y_train
				// A stores current activations computed from forward prop for all m samples
				int inputSize = inputShape[0] * inputShape[1] * inputShape[2];
				int memsize = inputSize * batch_size;
				/*
				std::cout << "A_v: " << memsize << " " 
					<< "TOTAL pix:" <<inputSize << " " 
					<< "A.size:" << A.size() << " "
					<< "Batchsize:" << batch_size << std::endl;
				*/
				float* A_v = (float*) malloc(memsize * sizeof(float));
				memsize = CATEGORIES * batch_size;
				float* Y_one_hot = (float*) malloc(memsize * sizeof(float));
				vectorize(A, A_v, inputSize, Y_batch, Y_one_hot, CATEGORIES);

				// A_v now: batch_size * curN
				// transpose A_v to curN * batchSize
				transpose(A_v, batch_size, inputSize);

				float* A_final = forwardPropagation(A_v, this->layers);

				// final activations -> compute cost and grads
				// cross-entropy cost
				avgMiniBatchCost += computeCost(A_final, Y_one_hot);

				correctPred += correctPredict(A_final, Y_one_hot);
				// compute grad from loss functions
				float* dZ = computeLossGrad(A_final, Y_one_hot);

				// backward propgation
				backwardPropagation(dZ, this->layers);

				std::cout << "\r" << "Batch progress: " << (mIdx + 1) * BATCH_SIZE << "/" << X_train.size() << std::flush;
			}

			duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
			avgMiniBatchCost /= numOfMiniBatches;
			std::cout << std::endl;
			std::cout << "Cost after " << eIdx << " epoch: " << avgMiniBatchCost << std::endl;
			std::cout << "Correct Predicts after " << eIdx << " epoch: " << correctPred << "/" << X_train.size()  << std::endl;
			std::cout << "Epoch Training time is: " << duration << " sec" << std::endl;
			std::cout << DELIMITER << std::endl;
		}

		std::cout << "Training Complete" << std::endl;
		std::cout << DELIMITER << std::endl;
	}

	void CNN::evaluate(std::vector<float*>& X_test, std::vector<int>& Y_test) {
		int inputSize = inputShape[0] * inputShape[1] * inputShape[2];
		int test_size = X_test.size();
		int memsize = inputSize * test_size;
		float* A_v = (float*)malloc(memsize * sizeof(float));
		memsize = CATEGORIES * test_size;
		float* Y_one_hot = (float*)malloc(memsize * sizeof(float));
		vectorize(X_test, A_v, inputSize, Y_test, Y_one_hot, CATEGORIES);

		// A_v now: batch_size * curN
		// transpose A_v to curN * batchSize
		transpose(A_v, test_size, inputSize);
		float* A_final = forwardPropagation(A_v, this->layers);
		float cost = computeCost(A_final, Y_one_hot);
		int correctPred = correctPredict(A_final, Y_one_hot);

		std::cout << "Final cost on test: " << cost << std::endl;
		std::cout << "Predict accuracy on test: " << correctPred << "/" << X_test.size() << std::endl;
	}

	// Relu Layer
	float* ReLU::forward(float* A_prev) {
		// allocate memory for current layer activation
		int curN = curShape[0];
		int prevN = prevShape[0];
		float *AL = (float *)malloc(curN * BATCH_SIZE * sizeof(float));
		if (this->A) {
			free(this->A);
			this->A = nullptr;
		}
		this->A = A_prev;

		// A_prev: prevN * batchSize
		// W: curN * prevN
		// AL: curN * batchSize
		matrixMul(AL, this->W, A_prev, curN, prevN, BATCH_SIZE);
		matrixAddBias(AL, this->b, curN, BATCH_SIZE);
		relu(AL, curN * BATCH_SIZE);
		return AL;
	}

	float* ReLU::backward(float* dZ) {
		float* dZ_prev, *dW, *dB, *Wtmp;
		int curN = curShape[0];
		int prevN = prevShape[0];

		dW = (float*)malloc(curN * prevN * sizeof(float));
		dB = (float*)malloc(curN * sizeof(float));
		dZ_prev = (float*)malloc(prevN * BATCH_SIZE * sizeof(float));
		Wtmp = (float*)malloc(prevN * curN * sizeof(float));
		memcpy(Wtmp, this->W, prevN *curN * sizeof(float));

		transpose(this->A, prevShape[0], BATCH_SIZE);
		matrixMul(dW, dZ, this->A, curN, BATCH_SIZE, prevN);
		// dZ: curN x batch_size
		matrixSum(dB, dZ, curN, BATCH_SIZE);

		// compute dZ_prev
		transpose(Wtmp, curN, prevN);
		matrixMul(dZ_prev, Wtmp, dZ, prevN, curN, BATCH_SIZE);
		reluGradFilter(dZ_prev, prevN, BATCH_SIZE);

		updateW(this->W, dW, curN, prevN, this->learningRate);
		updateB(this->b, dB, curN, this->learningRate);

		free(dW);
		free(dB);
		free(Wtmp);
		return dZ_prev;
	}

	// SoftMax Layer
	float* SoftMax::forward(float* A_prev) {
		// allocate memory for current layer activation
		int curN = curShape[0];
		int prevN = prevShape[0];
		float *AL = (float *)malloc(curN * BATCH_SIZE * sizeof(float));
		if (this->A) {
			free(this->A);
			this->A = nullptr;
		}
		this->A = A_prev;

		// A_prev: prevN * batchSize
		// W: curN * prevN
		// AL: curN * batchSize
		matrixMul(AL, this->W, A_prev, curN, prevN, BATCH_SIZE);
		matrixAddBias(AL, this->b, curN, BATCH_SIZE);
		softmax(AL, curN * BATCH_SIZE);
		return AL;
	}

	float* SoftMax::backward(float* dZ) {
		float* dZ_prev, *dW, *dB, *Wtmp;
		int curN = curShape[0];
		int prevN = prevShape[0];

		dW = (float*)malloc(curN * prevN * sizeof(float));
		dB = (float*)malloc(curN * sizeof(float));
		dZ_prev = (float*)malloc(prevN * BATCH_SIZE * sizeof(float));
		Wtmp = (float*)malloc(prevN * curN * sizeof(float));
		memcpy(Wtmp, this->W, prevN *curN * sizeof(float));

		transpose(this->A, prevShape[0], BATCH_SIZE);
		matrixMul(dW, dZ, this->A, curN, BATCH_SIZE, prevN);
		// dZ: curN x batch_size
		matrixSum(dB, dZ, curN, BATCH_SIZE);

		// compute dZ_prev
		transpose(Wtmp, curN, prevN);
		matrixMul(dZ_prev, Wtmp, dZ, prevN, curN, BATCH_SIZE);
		reluGradFilter(dZ_prev, prevN, BATCH_SIZE);

		updateW(this->W, dW, curN, prevN, this->learningRate);
		updateB(this->b, dB, curN, this->learningRate);
		
		free(dW);
		free(dB);
		free(Wtmp);
		return dZ_prev;
	}

	float CNN::computeCost(float* A_final, float* Y_one_hot) {
		float cost = crossEntropyLoss(A_final, Y_one_hot, CATEGORIES, BATCH_SIZE);
		return cost;
	}

	// compute grad from loss function
	// implement cross-entropy loss
	float* CNN::computeLossGrad(float* A_final, float* Y_one_hot) {
		float* dZ = elementWiseMinus(A_final, Y_one_hot, CATEGORIES, BATCH_SIZE);
		return dZ;
	}

	int CNN::correctPredict(float* A_final, float* Y_one_hot) {
		int count = getCorrectCount(A_final, Y_one_hot, CATEGORIES, BATCH_SIZE);
		return count;
	}
}
