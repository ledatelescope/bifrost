/*! \file testfft.cu
 *  \brief This file tests the fft.cu functionality.
 */

#include "fft.cu"
#include <gtest/gtest.h>
#include <bifrost/common.h>

/*! \brief Simple test of 2 dimensional real data
 *  through a forward bfFFT.
 */
TEST(FFTTest, Handles2dReal)
{
    /// Set up the test with a 3x2 element real array
    BFarray dataAndDescription;
    BFarray outputDataAndDescription;
    cufftReal hostData[3][2] = 
        {{1,2},{2,3},{3,4}};
    cufftReal** deviceData;
    cufftComplex outputHostData[3][2] = {};
    cufftComplex** outputDeviceData;
    /// Load the test array to the GPU
    cudaMalloc((void**)&deviceData, sizeof(cufftReal)*6);
    cudaMalloc((void**)&outputDeviceData, sizeof(cufftComplex)*6);
    cudaMemcpy(
        deviceData, hostData, 
        sizeof(cufftReal)*6, cudaMemcpyHostToDevice);
    /// Describe our input and output data
    dataAndDescription.data = deviceData;
    dataAndDescription.space = BF_SPACE_CUDA;
    dataAndDescription.shape[0] = 3;
    dataAndDescription.shape[1] = 2;
    dataAndDescription.dtype = 0;
    dataAndDescription.ndim = 2;
    dataAndDescription.strides[0] = 2*sizeof(cufftReal);
    dataAndDescription.strides[1] = sizeof(cufftReal);
    outputDataAndDescription = dataAndDescription;
    outputDataAndDescription.data = outputDeviceData;
    outputDataAndDescription.dtype = 1;
    outputDataAndDescription.strides[0] = 2*sizeof(cufftComplex);
    outputDataAndDescription.strides[1] = sizeof(cufftComplex);

    /// Perform the forward FFT in the separate output object
    bfFFT(&dataAndDescription, &outputDataAndDescription, FFT_FORWARD);
    /// Download the data back from the GPU
    cudaMemcpy(
        outputHostData, (cufftComplex*)outputDataAndDescription.data, 
        sizeof(cufftComplex)*6, cudaMemcpyDeviceToHost);

    /// Did the FFT produce the correct result?
    /// We check this only with one value.
    EXPECT_EQ(cuCrealf(outputHostData[0][0]),15);
}

/*! \brief Simple test of 1 dimensional real data
 *  through a forward bfFFT.
 */
TEST(FFTTest, Handles1dReal)
{
    /// Set up the test with a 4 element real array
    BFarray dataAndDescription;
    BFarray outputDataAndDescription;
    cufftReal hostData[4] = {1,3,6,4.5};
    cufftReal* deviceData;
    cufftComplex outputHostData[3];
    cufftComplex* outputDeviceData;
    /// Load this array to the GPU
    cudaMalloc((void**)&deviceData, sizeof(cufftReal)*5);
    cudaMalloc((void**)&outputDeviceData, sizeof(cufftComplex)*3);
    cudaMemcpy(
        deviceData, hostData, 
        sizeof(cufftReal)*4, cudaMemcpyHostToDevice);
    /// Describe our input and output data
    dataAndDescription.data = deviceData;
    dataAndDescription.space = BF_SPACE_CUDA;
    dataAndDescription.shape[0] = 4;
    dataAndDescription.dtype = 0;
    dataAndDescription.ndim = 1;
    dataAndDescription.strides[0] = sizeof(cufftReal);
    /// Output data is a different type, so requires
    /// a separate output object
    outputDataAndDescription = dataAndDescription;
    outputDataAndDescription.data = outputDeviceData;
    outputDataAndDescription.dtype = 1;
    outputDataAndDescription.strides[0] = sizeof(cufftComplex);

    /// Perform the forward FFT into the separate output data
    bfFFT(&dataAndDescription, &outputDataAndDescription, FFT_FORWARD);
    /// Download the data back from the GPU
    cudaMemcpy(
        outputHostData, (cufftComplex*)outputDataAndDescription.data, 
        sizeof(cufftComplex)*3, cudaMemcpyDeviceToHost);

    /// Did the FFT produce the correct result?
    /// We check this only with one value.
    EXPECT_EQ((int)cuCimagf(outputHostData[1]), 1);
}

/*! \brief Simple test of 2 dimensional complex data
 *  through a forward bfFFT.
 */
TEST(FFTTest, Handles2dComplex)
{
    /// Set up the test with a 3x3 element complex array
    BFarray dataAndDescription;
    cufftComplex hostData[3][3] = 
        {{{5,1},{0,0},{100,0}},
        {{5,1},{30,0},{100,0}},
        {{30,0},{0,0},{10,1}}};
    /// Load this array to the GPU
    cufftComplex** deviceData;
    cudaMalloc((void**)&deviceData, sizeof(cufftComplex)*9);
    cudaMemcpy(
        deviceData, hostData, 
        sizeof(cufftComplex)*9, cudaMemcpyHostToDevice);
    dataAndDescription.data = deviceData;
    /// Describe our data
    dataAndDescription.space = BF_SPACE_CUDA;
    dataAndDescription.shape[0] = 3;
    dataAndDescription.shape[1] = 3;
    dataAndDescription.dtype = 1;
    dataAndDescription.ndim = 2;
    dataAndDescription.strides[0] = 3*sizeof(cufftComplex);
    dataAndDescription.strides[1] = sizeof(cufftComplex);

    /// Perform the forward FFT in place
    bfFFT(&dataAndDescription, &dataAndDescription, FFT_FORWARD);
    /// Dowload data back from the GPU
    cudaMemcpy(
        hostData, (cufftComplex**)dataAndDescription.data, 
        sizeof(cufftComplex)*9, cudaMemcpyDeviceToHost);

    /// Did the FFT produce the correct result?
    /// We check this only with one value.
    EXPECT_EQ((int)cuCimagf(hostData[2][2]), -125);
}

/*! \brief Simple test of 1 dimensional complex data
 *  through an inverse bfFFT.
 */
TEST(FFTTest, InverseC2C)
{
    /// Set up the test with a 5 element complex array
    BFarray dataAndDescription;
    cufftComplex hostData[5] = {{0,-1},{0,0},{100,-100},{30,0},{-5,0}};
    /// Load this array to the GPU
    cufftComplex* deviceData;
    cudaMalloc((void**)&deviceData, sizeof(cufftComplex)*5);
    cudaMemcpy(deviceData, hostData, sizeof(cufftComplex)*5, cudaMemcpyHostToDevice);
    dataAndDescription.data = deviceData;
    /// Describe our data
    dataAndDescription.space = BF_SPACE_CUDA;
    dataAndDescription.shape[0] = 5;
    dataAndDescription.dtype = 1;
    dataAndDescription.ndim = 1;
    dataAndDescription.strides[0] = sizeof(cufftComplex);

    /// Perform the inverse FFT in place 
    bfFFT(&dataAndDescription, &dataAndDescription, FFT_INVERSE);
    /// Download back from the GPU
    cudaMemcpy(
        hostData, (cufftComplex*)dataAndDescription.data, 
        sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);

    /// Did the FFT produce the correct answer?
    EXPECT_EQ((int)cuCrealf(hostData[3]),139);
}

/*! \brief Simple test of 1 dimensional complex data
 *  through a forward bfFFT.
 */
TEST(FFTTest, Handles1dComplex)
{
    /// Set up the test with a 5 element complex array
    BFarray dataAndDescription;
    cufftComplex hostData[5] = 
        {{0,0},{30,0},{100,0},{30,0},{-5,0}};
    /// Load this array to the GPU
    cufftComplex* deviceData;
    cudaMalloc((void**)&deviceData, sizeof(cufftComplex)*5);
    cudaMemcpy(deviceData, hostData, sizeof(cufftComplex)*5, cudaMemcpyHostToDevice);
    dataAndDescription.data = deviceData;
    /// Describe our data
    dataAndDescription.space = BF_SPACE_CUDA;
    dataAndDescription.shape[0] = 5;
    dataAndDescription.dtype = 1;
    dataAndDescription.ndim = 1;
    dataAndDescription.strides[0] = sizeof(cufftComplex);

    /// Perform the forward FFT in place 
    bfFFT(&dataAndDescription, &dataAndDescription, FFT_FORWARD);
    /// Download back from the GPU.
    cudaMemcpy(
        hostData, (cufftComplex*)dataAndDescription.data, 
        sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);

    /// Test a single value of the output, to test for
    /// the FFT's correct answer?
    EXPECT_EQ((int)cuCimagf(hostData[4]),(int)74.431946);
}

// TODO: Add task functionality for rings?
// TODO: Add test for multiple GPUs.
// TODO: Add test for type conversion.

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
