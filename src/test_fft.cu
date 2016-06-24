/*! \file testfft.cu
 *  \brief This file tests the fft.cu functionality.
 */

#include "fft.cu"
#include <gtest/gtest.h>
#include <bifrost/common.h>

TEST(FFTTest, Handles2dReal)
{
    BFarray my_data;
    BFarray out_data;
    BFreal set_data[3][2] = 
        {{1,2},{2,3},{3,4}};
    BFreal** some_data;
    BFcomplex* odata;
    cudaMalloc((void**)&some_data, sizeof(BFreal)*6);
    cudaMalloc((void**)&odata, sizeof(BFcomplex)*6);
    cudaMemcpy(
        some_data, set_data, 
        sizeof(BFreal)*6, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 3;
    my_data.shape[1] = 2;
    my_data.dtype = 0;
    my_data.ndim = 2;
    my_data.strides[0] = 2*sizeof(BFreal);
    my_data.strides[1] = sizeof(BFreal);
    out_data = my_data;
    out_data.data = odata;
    out_data.dtype = 1;
    out_data.strides[0] = 2*sizeof(BFcomplex);
    out_data.strides[1] = sizeof(BFcomplex);
    bfFFT(&my_data, &out_data, FFT_FORWARD);
    cufftComplex localdata[3][2] = {};
    cudaMemcpy(
        localdata, (cufftComplex*)out_data.data, 
        sizeof(cufftComplex)*6, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cuCrealf(localdata[0][0]),15);
}

TEST(FFTTest, Handles1dReal)
{
    BFarray my_data;
    BFarray out_data;
    BFreal set_data[4] = {1,3,6,4.5};
    BFreal* some_data;
    BFcomplex* odata;
    cudaMalloc((void**)&some_data, sizeof(BFreal)*5);
    cudaMalloc((void**)&odata, sizeof(BFcomplex)*3);
    cudaMemcpy(
        some_data, set_data, 
        sizeof(BFreal)*4, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 4;
    my_data.dtype = 0;
    my_data.ndim = 1;
    my_data.strides[0] = sizeof(BFreal);
    out_data = my_data;
    out_data.data = odata;
    out_data.dtype = 1;
    out_data.strides[0] = sizeof(BFcomplex);
    bfFFT(&my_data, &out_data, FFT_FORWARD);
    cufftComplex localdata[3] = {};
    cudaMemcpy(
        localdata, (cufftComplex*)out_data.data, 
        sizeof(cufftComplex)*3, cudaMemcpyDeviceToHost);
    EXPECT_EQ((int)cuCimagf(localdata[1]), 1);
}

TEST(FFTTest, Handles2dComplex)
{
    BFarray my_data;
    BFcomplex set_data[3][3] = 
        {{{5,1},{0,0},{100,0}},
        {{5,1},{30,0},{100,0}},
        {{30,0},{0,0},{10,1}}};
    BFcomplex** some_data;
    cudaMalloc((void**)&some_data, sizeof(BFcomplex)*9);
    cudaMemcpy(
        some_data, set_data, 
        sizeof(BFcomplex)*9, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 3;
    my_data.shape[1] = 3;
    my_data.dtype = 1;
    my_data.ndim = 2;
    my_data.strides[0] = 3*sizeof(BFcomplex);
    my_data.strides[1] = sizeof(BFcomplex);
    bfFFT(&my_data, &my_data, FFT_FORWARD);
    cufftComplex localdata[3][3]={};
    cudaMemcpy(
        localdata, (cufftComplex**)my_data.data, 
        sizeof(cufftComplex)*9, cudaMemcpyDeviceToHost);
    EXPECT_EQ((int)cuCimagf(localdata[2][2]), -125);
}

TEST(FFTTest, InverseC2C)
{
    BFarray my_data;
    BFcomplex set_data[5] = {{0,-1},{0,0},{100,-100},{30,0},{-5,0}};
    BFcomplex* some_data;
    cudaMalloc((void**)&some_data, sizeof(BFcomplex)*5);
    cudaMemcpy(some_data, set_data, sizeof(BFcomplex)*5, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 5;
    my_data.dtype = 1;
    my_data.ndim = 1;
    my_data.strides[0] = sizeof(BFcomplex);
    bfFFT(&my_data, &my_data, FFT_INVERSE);
    cufftComplex localdata[5]={};
    cudaMemcpy(localdata, (cufftComplex*)my_data.data, sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);

    EXPECT_EQ((int)cuCrealf(localdata[3]),139);
}

TEST(FFTTest, Handles1dComplex)
{
    //Set up the test with a 5 element complex array
    BFarray my_data;
    BFcomplex set_data[5] = {{0,0},{30,0},{100,0},{30,0},{-5,0}};
    BFcomplex* some_data;
    cudaMalloc((void**)&some_data, sizeof(BFcomplex)*5);
    cudaMemcpy(some_data, set_data, sizeof(BFcomplex)*5, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 5;
    my_data.dtype = 1;
    my_data.ndim = 1;
    my_data.strides[0] = sizeof(BFcomplex);

    //Perform the FFT.
    bfFFT(&my_data, &my_data, FFT_FORWARD);
    //Unload the data from the device
    cufftComplex localdata[5]={};
    cudaMemcpy(
        localdata, (cufftComplex*)my_data.data, 
        sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);

    //Assert that last imaginary value is about 74
    EXPECT_EQ((int)cuCimagf(localdata[4]),(int)74.431946);
}



int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
