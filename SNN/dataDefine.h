#pragma once
#include<immintrin.h>
#include<iostream>
#include<random>
#include<fstream>
#include<thread>
#include<mutex>
#define MNISTDIM 28
#define TESTNUM 10000

#define AlignBytes 32 
#define AlignBytesPool 32 
#define AlignBytes16 16 
#define AlignVec(x,V) ((x)%(V)==0?(x):((x)/(V)+1)*(V))

#define MNISTBLOCK AlignVec(MNISTDIM*MNISTDIM,AlignBytes/sizeof(float))

//#define TIMESTEP 25
#define OUTCLASS 10
#define TAO 100
#define NOBACKGGROUNDB
#define PI 3.1415926f
#define TRUCATE 100000
#define LAYER1 784
#define VTHR 5
static std::default_random_engine gen;

typedef struct dim
{
	int dim1;
	int dim2;
	int dim3;
	dim()
	{
		dim1 = 0;
		dim2 = 0;
		dim3 = 0;
	}
	dim(int a, int b, int c)
	{
		dim1 = a;
		dim2 = b;
		dim3 = c;
	}
}dim;
enum OptiMethod
{
	NooP = 0,
	SGD,
	SGDM,
	SGDMW,
	SGNAD,
	ADAM,
	ADAMW
};
typedef struct Optimizer
{
	OptiMethod method;
	float beta1;
	float beta2;
	float sigma;
	float l2Lamda;
	Optimizer()
	{
		beta1 = 0.9f;
		beta2 = 0.99f;
		sigma = 0.000001f;
		method = NooP;
		l2Lamda = 0.0;
	}
}Optimizer;
class tensor
{
public:
	tensor() ;
	~tensor() ;
	 
	void initData(dim initDim) ;
	void freeData();
	
//	void randInitSpike(float sd, float mu);
	void randInit(float sd, float mu);
	dim getDim() { return tensorDim; };
	float* getData() { return data; };
	size_t getBlockSize() { return blocksize;  };
	size_t getDim23BlockSize() { return SingleBlocksizeDim23; };
	float* getDim3Data(int d1, int d2);
	float* getDim23Data(int d1);
	/*
		math operation
	*/
	void applyRatio(float ra, int t = -1, int b = 0);
	void addValue(float ra, int t = -1, int b = 0);
	void valueInit(float value, int t = -1, int b = 0);
	void divedValue(float ra, int t = -1, int b = 0);
	void sqrtTensor(int t, int b);
	void mulTensor(tensor ts, int t = -1, int b = 0);
	void divTensor(tensor ts, int t = -1, int b = 0);
	void copyTensor(tensor ts, int t = -1, int b = 0);
	void addTensor(tensor ts, int t = -1, int b = 0);
	void truncateValue(float ra, int t = -1, int b = 0);

	void recordT(int b);
	void recordW(int b);
private:
	dim tensorDim;//time x batch x feature length
	float* data;
	
	size_t	blocksize;
	size_t	SingleBlocksizeDim3;
	size_t	SingleBlocksizeDim23;
};