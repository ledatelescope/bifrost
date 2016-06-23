/*! \file testfft.cu
 *  \brief This file tests the fft.cu functionality.
 */

#include "fft.cu"
#include <assert.hpp>

BFstatus test_bffft_real_2d()
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
    if (bfFFT(&my_data, &out_data, FFT_FORWARD) != BF_STATUS_SUCCESS)
    {
        return BF_STATUS_INTERNAL_ERROR; 
    }
    cufftComplex localdata[3][2] = {};
    cudaMemcpy(
        localdata, (cufftComplex*)out_data.data, 
        sizeof(cufftComplex)*6, cudaMemcpyDeviceToHost);
    BF_ASSERT(cuCrealf(localdata[0][0])==15.2,0);
    printf("Still good?");
    return BF_STATUS_SUCCESS;
}
void test_bffft_real()
{
    BFarray my_data;
    BFarray out_data;
    BFreal set_data[4] = {1,3,6,2.5134};
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
    if (bfFFT(&my_data, &out_data, FFT_FORWARD) != BF_STATUS_SUCCESS)
    {
        printf("bfFFT failed!\n");
        return; 
    }
    cufftComplex localdata[3] = {};
    cudaMemcpy(
        localdata, (cufftComplex*)out_data.data, 
        sizeof(cufftComplex)*3, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 3; i++)
        printf("%f+I%f\n",cuCrealf(localdata[i]),cuCimagf(localdata[i]));
    return;
}

void test_bffft_2d()
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
    if (bfFFT(&my_data, &my_data, FFT_FORWARD) != BF_STATUS_SUCCESS)
    {
        printf("bfFFT failed!\n");
        return; 
    }
    cufftComplex localdata[3][3]={};
    cudaMemcpy(
        localdata, (cufftComplex**)my_data.data, 
        sizeof(cufftComplex)*9, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
                printf("%f\n",cuCrealf(localdata[i][j]));
    }
    //print successfully fft'd data.
    return;
}

void test_bffft_inverse_1d()
{
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
    bfFFT(&my_data, &my_data, FFT_INVERSE);
    cufftComplex localdata[5]={};
    cudaMemcpy(localdata, (cufftComplex*)my_data.data, sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++)
        printf("%f+I%f\n",cuCrealf(localdata[i]),cuCimagf(localdata[i]));
    //print successfully fft'd data.
}

void test_bffft_1d()
{
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
    bfFFT(&my_data, &my_data, FFT_FORWARD);
    cufftComplex localdata[5]={};
    cudaMemcpy(localdata, (cufftComplex*)my_data.data, sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++)
        printf("%f+I%f\n",cuCrealf(localdata[i]),cuCimagf(localdata[i]));
    //print successfully fft'd data.
}


int main()
{
    printf("Running...\n");
    //test_bffft_1d();
    //test_bffft_2d();
    //test_bffft_real();
    //test_bffft_inverse_1d();
    test_bffft_real_2d();
    printf("Done\n");
    return 0;
}
