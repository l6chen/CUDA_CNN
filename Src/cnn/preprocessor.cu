/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "preprocessor.h"
#define W_r 224
#define H_r 224

//#include "narray.h"

namespace prep {
	//using namespace narray;
	const int NUMBER_OF_TRAINING_SAMPLES = 1500;
	const int NUMBER_OF_TEST_SAMPLES = 1306;
	const std::string DELIMITER = "================================================================================";

	// image data lays out as a 2d matrix, with each pixel contains 3 vals {blue, green, red}
	//void readImageData(std::vector<Narray<float>*>& vector, std::vector<std::string> imageName, std::string imageDir) {
	void readImageData(std::vector<float*>& vector, std::vector<std::string> imageName, std::string imageDir, int *imgHeight, int *imgWidth, int* imgChannels) {
		int count = 0;
		for (std::string image : imageName) {
			cv::Mat img_ori = cv::imread(imageDir + "/" + image);
			cv::Size dsize = cv::Size(W_r, H_r);
			cv::Mat img = cv::Mat(dsize, img_ori.type());
			cv::resize(img_ori, img, dsize);
			if (!img.data) {
				std::cout << "issue in reading images" << std::endl;
				exit(1);
			}
			int channels = img.channels();
			int size = img.total() * channels;
			//std::cout << "H:" << img.rows << " W:" << img.rows << " C:" << channels << std::endl;
			//std::cout << "imgtotal:" << img.total() << std::endl;

			if (imgHeight && imgWidth && imgChannels && imgHeight != 0) {
				*imgHeight = img.rows;
				*imgWidth = img.cols;
				*imgChannels = channels;
			}
			
			float* data = new float[size];
			for (int i = 0; i < size; ++i) {
				// normalize the data
				data[i] = (float)img.data[i] / 255.0;
			}
			//Narray<float>* arr = new Narray<float>(data, size, new int[3] {img.rows, img.cols, channels}, 3);
			//vector.push_back(arr);
			vector.push_back(data);
			++count;
			if (count % 100 == 0) {
				std::cout << "\r" << count << " Images loaded" << std::flush;
			}
		}

		std::cout << std::endl;
	}
	void readData(std::string fileName, std::vector<int>& vector, int samples) {
		std::fstream fs(fileName.c_str(), std::fstream::in);
		int count = 0;
		if (fs.good()) {
			int tmp;
			while (fs >> tmp && count < samples) {
				// normalize label to [0, 15]
				vector.push_back(tmp - 1);
				++count;
			}
		}
		fs.close();
	}

	void readData(std::string fileName, std::vector<std::string>& vector, int samples) {
		std::fstream fs(fileName.c_str(), std::fstream::in);
		int count = 0;
		if (fs.good()) {
			std::string tmp;
			while (std::getline(fs, tmp) && count < samples) {
				vector.push_back(tmp);
				++count;
			}
		}
		fs.close();
	}

	void Preprocessor::loadDataSets() {
		// Constants for file locations
		std::string lists = "/lists";
		std::string images = "/images/Images";
		std::string train_path = this->resourceDir + lists + "/train_list.txt";
		std::string train_labels_path = this->resourceDir + lists + "/train_labels.txt";
		std::string test_path = this->resourceDir + lists + "/test_list.txt";
		std::string test_labels_path = this->resourceDir + lists + "/test_labels.txt";
		std::string imgDir = this->resourceDir + images;
		std::cout << DELIMITER << std::endl;
		// Use STL container to contain two label lists
		std::vector<std::string> X_train_path;
		std::vector<std::string> X_test_path;

		readData(train_path, X_train_path, NUMBER_OF_TRAINING_SAMPLES);
		readData(test_path, X_test_path, NUMBER_OF_TEST_SAMPLES);
		readData(train_labels_path, this->Y_train_labels, NUMBER_OF_TRAINING_SAMPLES);
		readData(test_labels_path, this->Y_test_labels, NUMBER_OF_TEST_SAMPLES);

		// read in image data
 		std::cout << "Start Loading Training Images:" << std::endl;
		readImageData(this->X_train, X_train_path, imgDir, &(this->imgHeight), &(this->imgWidth), &(this->imgChannels));
		std::cout << "Complete" << std::endl;

		std::cout << DELIMITER << std::endl;

		std::cout << "Start Loading Test Images:" << std::endl;
		readImageData(this->X_test, X_test_path, imgDir, nullptr, nullptr, nullptr);
		std::cout << "Complete" << std::endl;
	}
}