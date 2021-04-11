#pragma once
#include "Imagelib.h"

// Tensor3D는 크기가 (nH x nW x nC)인 3차원 tensor를 관리함

class Tensor3D {
private:
	double*** tensor;
	int nH; // height
	int nW; // width
	int nC; // channel
public:
	Tensor3D(int _nH, int _nW, int _nC) : nH(_nH), nW(_nW), nC(_nC) {
		// 동작: 1)3차원 행렬을 동적할당하여, tensor에 시작 주소값을 저장
		//       2)그 후 모든 element의 값을 0으로 초기화
		// 사용함수: dmatrix3D(): 3차원 행렬을 동적 할당해서 pointer를 반환하는 함수

		tensor = dmatrix3D(_nH, _nW, _nC);
	}
	~Tensor3D() {
		// 동작: 3차원 동적 배열인 tensor를 할당 해제
		// 사용함수: free_dmatrix3D(): 3차원 동적 할당된 행렬을 할당 해제하는 함수
		free_dmatrix3D(tensor, nH, nW, nC);

	}
	void set_elem(int _h, int _w, int _c, double _val) { tensor[_h][_w][_c] = _val; }
	double get_elem(int _h, int _w, int _c)	const {
		// 동작: 행=_h, 열= _w, 채널= _c 위치 element를 반환할 것
		return tensor[_h][_w][_c];
	}

	void get_info(int& _nH, int& _nW, int& _nC) const {
		// 동작: 행렬의 차원(nH, nW, nC)을 pass by reference로 반환
		 _nH = nH;
		 _nW = nW;
		 _nC = nC;
	}

	void set_tensor(double*** _tensor) { tensor = _tensor; }
	double*** get_tensor() const { return tensor; }

	void print() const {
		// 동작: 행렬의 크기 (nH*nW*nC)를 화면에 출력
		cout << nH << "*" << nW << "*" << nC << endl;
	}
};