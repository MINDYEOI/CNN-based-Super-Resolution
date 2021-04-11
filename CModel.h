#pragma once
#include "CLayer.h"

// Model은 layer와 tensor들을 모두 통합 관리하여 효과적으로 CNN이 수행될 수 있도록 함

class Model {
private:
	vector<Layer*> layers; //layer들을 순차적으로 저장
	vector<Tensor3D*> tensors;// tensor들을 순차적으로 저장 ( 0번째 tensor는 0번째 layer의 입력, 마찬가지로 1번째 tensor는 1번째 layer의 입력이자 0번째 layer의 출력임)
public:
	Model() {}
	void add_layer(Layer* layer) {
		// 동작: layer 객체를 layers vector의 마지막 element로 저장 
		layers.push_back(layer);
	}
	~Model() {
	//	// 동작: layers와 tensors의 모든 element를 동적할당 해제해 줄 것
		for (auto elemL : layers)
		{
			delete elemL;
		}
	
		for (auto elemT : tensors)
		{
			delete elemT;
		}

	}
	void test(string filename_input, string filename_output) {
		// (구현할 것)
		// 동작1: filename_input으로부터 이미지를 읽어와서, tensor로 변환한 다음 CNN을 수행한다음 그 결과물을 filename_output에 저장
		// 동작2: 주석 (1), (2), (3), (4) 중 (2)번만 구현하면 됨 

		int nH, nW;
		double** input_img_Y, ** input_img_U, ** input_img_V;
		byte* pLoadImage;

		// (1) 영상을 읽어서 2차원 배열로 저장 (input_img_Y, U, V는 read_image에서 동적 할당됨)
		read_image(filename_input, pLoadImage, input_img_Y, input_img_U, input_img_V, nH, nW);
		cout << "Reading (" << filename_input << ") is complete..." << endl;


		// (2) 이부분만 구현할 것
		// 동작1: 현재 tensors의 0번째 element에 영상(CNN의 입력)이 이미 저장되어 있음
		// 동작2: tensors vector의 i번째 tensor를 layers vector에 있는 i번째 layer의 forward함수로 입력받고, 그 결과를 tensors vector의 i+1번째 tensor로 저장함
		// 동작3: 결과적으로 tensors의 가장 마지막 tensor는 CNN의 출력값이 됨 (이 출력값은 (3)에서 1차원 배열로 변환되어 이미지 파일에 저장됨 

		for (int i = 0; i < layers.size(); i++)
		{
			tensors.push_back(layers[i]->forward(tensors[i]));
		}
		


		cout << "Super-resolution is complete..." << endl;

		// (3) CNN의 출력(마지막 tensor)을 2차원 배열로 변환 후 U, V 채널과 함께 이미지로 저장
		Tensor3D* output_tensor_Y = tensors.at(tensors.size() - 1);
		save_image(filename_output, pLoadImage, output_tensor_Y, input_img_U, input_img_V, nH, nW);
		cout << "Saving (" << filename_output << ") is complete..." << endl;

		// (4) 할당 해제
		free(pLoadImage);
		free_dmatrix2D(input_img_Y, nH, nW);
		free_dmatrix2D(input_img_U, nH, nW);
		free_dmatrix2D(input_img_V, nH, nW);
	}

	void read_image(const string filename, byte*& pLoadImage, double**& img_Y, double**& img_U, double**& img_V, int& nH, int& nW) {

		LoadBmp(filename.c_str(), &pLoadImage, nH, nW);///이미지파일 읽기

		img_Y = dmatrix2D(nH, nW);
		img_U = dmatrix2D(nH, nW);
		img_V = dmatrix2D(nH, nW);

		convert1Dto2D(pLoadImage, img_Y, img_U, img_V, nH, nW);

		// 입력 영상을 tensor로 변환 후 첫번째 element에 저장
		double*** inImage3D = dmatrix3D(nH, nW, 1);
		convert2Dto3D(img_Y, inImage3D, nH, nW);

		Tensor3D* temp = new Tensor3D(nH, nW, 1);
		temp->set_tensor(inImage3D);
		tensors.push_back(temp);

	}
	void save_image(string filename, byte*& pLoadImage, Tensor3D*& tensor_Y, double** img_U, double** img_V, int nH, int nW) {
		double** img_Y = dmatrix2D(nH, nW);
		convert3Dto2D(tensor_Y->get_tensor(), img_Y, nH, nW);
		convert2Dto1D(img_Y, img_U, img_V, pLoadImage, nH, nW);
		SaveBmp(filename.c_str(), pLoadImage, nH, nW);
		free_dmatrix2D(img_Y, nH, nW);
	}
	void print_layer_info() const {
		cout << endl << "(Layer information)_____________" << endl;
		for (unsigned i = 0; i < layers.size(); i++) {
			cout << i + 1 << "-th layer: ";
			layers.at(i)->print();
		}
	}
	void print_tensor_info() const {
		cout << endl << "(Tensor information)_____________" << endl;
		for (unsigned i = 0; i < tensors.size(); i++) {
			cout << i + 1 << "-th tensor: ";
			tensors.at(i)->print();
		}
	}

	//	void train();
};