/**
*	Author: Hou Wang
*   PID: A53241783
*/

#pragma once

#include <string>
#include <vector>
//#include "narray.h"

namespace prep {

	class Preprocessor {
	private:
		std::string resourceDir;
		//std::vector<narray::Narray<float>*> X_train;
		//std::vector<narray::Narray<float>*> X_test;
		std::vector<float*> X_train;
		std::vector<float*> X_test;
		std::vector<int> Y_train_labels;
		std::vector<int> Y_test_labels;
		int imgHeight;
		int imgWidth;
		int imgChannels;
	public:
		Preprocessor(std::string res) {
			resourceDir = res;
			imgHeight = 0;
			imgWidth = 0;
		}

		void loadDataSets();

		//std::vector<narray::Narray<float>*> getTrainingData() { return X_train; }
		//std::vector<narray::Narray<float>*> getTestData() { return X_test; }
		int getImgHeight() { return imgHeight; }
		int getImgWidth() { return imgWidth;  }
		int getImgChannels() { return imgChannels; };
		std::vector<float*> getTrainingData() { return X_train; }
		std::vector<float*> getTestData() { return X_test; }
		std::vector<int> getTrainingLabels() { return Y_train_labels;  }
		std::vector<int> getTestLabels() { return Y_test_labels;  }

	};
}