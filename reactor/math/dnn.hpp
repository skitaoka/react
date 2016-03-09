#pragma once

#ifndef REACTOR_MATH_DNN_HPP_INCLUDED
#define REACTOR_MATH_DNN_HPP_INCLUDED

#include <memory>
#include <vector>
#include <exception>

#include <cudnn.h>

namespace reactor {
  namespace cuda {
    inline void throw_if_failed(cudnnStatus_t status) {
      if (status) {
        throw ::cudnnGetErrorString(status);
      }
    }

    class cudnn : std::shared_ptr<std::remove_pointer_t<cudnnHandle_t>> {
      using super_type = std::shared_ptr<std::remove_pointer_t<cudnnHandle_t>>;

    public:
      cudnn()
      {
        cudnnHandle_t handle = nullptr;
        throw_if_failed(::cudnnCreate(&handle));
        super_type::reset(handle, [](cudnnHandle_t handle) {
          ::cudnnDestroy(handle);
        });
      }

      cudaStream_t stream() const {
        cudaStream_t streamId = nullptr;
        throw_if_failed(::cudnnGetStream(super_type::get(), &streamId));
        return streamId;
      }

      void set_stream(cudaStream_t streamId) const {
        throw_if_failed(::cudnnSetStream(super_type::get(), streamId));
      }

      /// y@yDesc = alpha * b@bDesc + beta * y@yDesc
      void transform_tensor(
        float alpha, cudnnTensorDescriptor_t const xDesc, float const* x,
        float beta, cudnnTensorDescriptor_t const yDesc, float* y) {
        throw_if_failed(::cudnnTransformTensor(super_type::get(), &alpha, xDesc, x, &beta, yDesc, y));
      }

      /// y@yDesc = alpha * b@bDesc + beta * y@yDesc
      void add_tensor(
        float alpha, cudnnTensorDescriptor_t bDesc, float const* b,
        float beta, cudnnTensorDescriptor_t yDesc, float* y) {
        throw_if_failed(::cudnnAddTensor_v3(super_type::get(), &alpha, bDesc, b, &beta, yDesc, y));
      }

      /// y@yDesc = alpha * b@bDesc + beta * y@yDesc
      void add_tensor(cudnnAddMode_t mode,
        float alpha, cudnnTensorDescriptor_t bDesc, float const* b,
        float beta, cudnnTensorDescriptor_t yDesc, float* y) {
        throw_if_failed(::cudnnAddTensor_v2(super_type::get(), mode, &alpha, bDesc, b, &beta, yDesc, y));
      }

      /// y@yDesc = value
      void set_tensor(cudnnTensorDescriptor_t const yDesc, float* y, float value) {
        throw_if_failed(::cudnnSetTensor(super_type::get(), yDesc, y, &value));
      }

      //
      // convolution forward
      //
      std::vector<cudnnConvolutionFwdAlgoPerf_t> find_convolution_forward_algorithm(
        cudnnTensorDescriptor_t const xDesc, cudnnFilterDescriptor_t const wDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnTensorDescriptor_t const yDesc,
        int requestedAlgoCount) {
        int returnedAlgoCount = 0;
        std::vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(requestedAlgoCount);
        throw_if_failed(::cudnnFindConvolutionForwardAlgorithm(super_type::get(), xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, perfResults.data()));
        perfResults.resize(returnedAlgoCount);
        return perfResults;
      }

      cudnnConvolutionFwdAlgo_t get_convolution_forward_algorithm(
        cudnnTensorDescriptor_t const xDesc, cudnnFilterDescriptor_t const wDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnTensorDescriptor_t const yDesc,
        cudnnConvolutionFwdPreference_t preference, std::size_t memoryLimitInBytes) {
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        throw_if_failed(::cudnnGetConvolutionForwardAlgorithm(super_type::get(), xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, &algo));
        return algo;
      }

      std::size_t get_convolution_forward_workspace_size(
        cudnnTensorDescriptor_t const xDesc, cudnnFilterDescriptor_t const wDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnTensorDescriptor_t const yDesc,
        cudnnConvolutionFwdAlgo_t algo) {
        std::size_t sizeInBytes = 0;
        throw_if_failed(::cudnnGetConvolutionForwardWorkspaceSize(super_type::get(), xDesc, wDesc, convDesc, yDesc, algo, &sizeInBytes));
        return sizeInBytes;
      }

      void convolution_forward(
        void const* alpha,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        cudnnFilterDescriptor_t const wDesc, void const* w,
        cudnnConvolutionDescriptor_t const convDesc,
        cudnnConvolutionFwdAlgo_t algo,
        void* workSpace, std::size_t workSpaceSizeInBytes,
        void const* beta,
        cudnnTensorDescriptor_t const yDesc, void* y) {
        throw_if_failed(::cudnnConvolutionForward(super_type::get(), alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y));
      }

      //
      // convolution backward filter
      //
      std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> find_convolution_backward_filter_algorithm(
        cudnnTensorDescriptor_t const xDesc, cudnnTensorDescriptor_t const dyDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnFilterDescriptor_t const dwDesc,
        int requestedAlgoCount) {
        int returnedAlgoCount = 0;
        std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perfResults(requestedAlgoCount);
        throw_if_failed(::cudnnFindConvolutionBackwardFilterAlgorithm(super_type::get(), xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, perfResults.data()));
        return perfResults;
      }

      cudnnConvolutionBwdFilterAlgo_t get_convolution_backward_filter_algorithm(
        cudnnTensorDescriptor_t const xDesc, cudnnTensorDescriptor_t const dyDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnFilterDescriptor_t const dwDesc,
        cudnnConvolutionBwdFilterPreference_t preference, std::size_t memoryLimitInBytes) {
        cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        throw_if_failed(::cudnnGetConvolutionBackwardFilterAlgorithm(super_type::get(), xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, &algo));
        return algo;
      }

      std::size_t get_convolution_backward_filter_workspace_size(
        cudnnTensorDescriptor_t const xDesc, cudnnTensorDescriptor_t dyDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnFilterDescriptor_t const dwDesc,
        cudnnConvolutionBwdFilterAlgo_t algo) {
        std::size_t sizeInBytes = 0;
        throw_if_failed(::cudnnGetConvolutionBackwardFilterWorkspaceSize(super_type::get(), xDesc, dyDesc, convDesc, dwDesc, algo, &sizeInBytes));
        return sizeInBytes;
      }

      // cudnnConvolutionBackwardFilter_v2 = cudnnConvolutionBackwardFilter_v3 with
      //   algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
      //   workSpace = nullptr
      //   workSpaceSizeInBytes = 0
      void convolution_backward_filter(
        void const* alpha,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        cudnnConvolutionDescriptor_t const convDesc,
        cudnnConvolutionBwdFilterAlgo_t algo,
        void* workSpace, std::size_t workSpaceSizeInBytes,
        void const* beta,
        cudnnFilterDescriptor_t const dwDesc, void* dw) {
        throw_if_failed(cudnnConvolutionBackwardFilter_v3(super_type::get(), alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw));
      }

      //
      // convolution backward data
      //
      std::vector<cudnnConvolutionBwdDataAlgoPerf_t> find_convolution_backward_data_algorithm(
        cudnnFilterDescriptor_t const wDesc, cudnnTensorDescriptor_t const dyDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnTensorDescriptor_t const dxDesc,
        int requestedAlgoCount) {
        int returnedAlgoCount = 0;
        std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perfResults(requestedAlgoCount);
        throw_if_failed(::cudnnFindConvolutionBackwardDataAlgorithm(super_type::get(), wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, &returnedAlgoCount, perfResults.data()));
        perfResults.resize(returnedAlgoCount);
        return perfResults;
      }

      cudnnConvolutionBwdDataAlgo_t get_convolution_backward_data_algorithm(
        cudnnFilterDescriptor_t const wDesc, cudnnTensorDescriptor_t const dyDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnTensorDescriptor_t const dxDesc,
        cudnnConvolutionBwdDataPreference_t preference, std::size_t memoryLimitInBytes) {
        cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        throw_if_failed(::cudnnGetConvolutionBackwardDataAlgorithm(super_type::get(), wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, &algo));
        return algo;
      }

      std::size_t get_convolution_backward_data_workspace_size(
        cudnnFilterDescriptor_t const wDesc, cudnnTensorDescriptor_t const dyDesc,
        cudnnConvolutionDescriptor_t const convDesc, cudnnTensorDescriptor_t const dxDesc,
        cudnnConvolutionBwdDataAlgo_t algo) {
        std::size_t sizeInBytes = 0;
        throw_if_failed(::cudnnGetConvolutionBackwardDataWorkspaceSize(super_type::get(), wDesc, dyDesc, convDesc, dxDesc, algo, &sizeInBytes));
        return sizeInBytes;
      }

      void convolution_backward_data(
        void const* alpha,
        cudnnFilterDescriptor_t const wDesc, void const* w,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        cudnnConvolutionDescriptor_t const convDesc,
        cudnnConvolutionBwdDataAlgo_t algo,
        void* workSpace, std::size_t workSpaceSizeInBytes,
        void const* beta,
        cudnnTensorDescriptor_t const dxDesc, void* dx) {
        throw_if_failed(::cudnnConvolutionBackwardData_v3(super_type::get(), alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx));
      }

      //
      //
      //
      void convolution_backward_bias(
        void const* alpha, cudnnTensorDescriptor_t const dyDesc, void const* dy,
        void const* beta, cudnnTensorDescriptor_t const dbDesc, void* db) {
        throw_if_failed(::cudnnConvolutionBackwardBias(super_type::get(), alpha, dyDesc, dy, beta, dbDesc, db));
      }

      void softmax_forward(
        cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
        void const* alpha,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta,
        cudnnTensorDescriptor_t const yDesc, void* y) {
        throw_if_failed(::cudnnSoftmaxForward(super_type::get(), algo, mode, alpha, xDesc, x, beta, yDesc, y));
      }

      void softmax_backward(
        cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
        void const* alpha,
        cudnnTensorDescriptor_t const yDesc, void const* y,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        void const* beta,
        cudnnTensorDescriptor_t const dxDesc, void* dx) {
        throw_if_failed(::cudnnSoftmaxBackward(super_type::get(), algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx));
      }

      void pooling_forward(
        cudnnPoolingDescriptor_t const poolingDesc,
        void const* alpha,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta,
        cudnnTensorDescriptor_t const yDesc, void* y) {
        throw_if_failed(::cudnnPoolingForward(super_type::get(), poolingDesc, alpha, xDesc, x, beta, yDesc, y));
      }

      void pooling_backward(
        cudnnPoolingDescriptor_t const poolingDesc,
        void const* alpha,
        cudnnTensorDescriptor_t const yDesc, void const* y,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta,
        cudnnTensorDescriptor_t const dxDesc, void* dx) {
        throw_if_failed(::cudnnPoolingBackward(super_type::get(), poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
      }

      void activation_forward(
        cudnnActivationMode_t mode,
        void const* alpha, cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta, cudnnTensorDescriptor_t const yDesc, void* y) {
        throw_if_failed(::cudnnActivationForward(super_type::get(), mode, alpha, xDesc, x, beta, yDesc, y));
      }

      void activation_forward(
        cudnnActivationDescriptor_t activationDesc,
        void const* alpha, cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta, cudnnTensorDescriptor_t const yDesc, void* y) {
        throw_if_failed(::cudnnActivationForward_v4(super_type::get(), activationDesc, alpha, xDesc, x, beta, yDesc, y));
      }

      void activation_backward(
        cudnnActivationMode_t mode,
        void const* alpha,
        cudnnTensorDescriptor_t const yDesc, void const* y,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta,
        cudnnTensorDescriptor_t const dxDesc, void* dx) {
        throw_if_failed(::cudnnActivationBackward(super_type::get(), mode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
      }

      void activation_backward(
        cudnnActivationDescriptor_t activationDesc,
        void const* alpha,
        cudnnTensorDescriptor_t const yDesc, void const* y,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta,
        cudnnTensorDescriptor_t const dxDesc, void* dx) {
        throw_if_failed(::cudnnActivationBackward_v4(super_type::get(), activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
      }

      void lrn_cross_channel_forward(
        cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
        void const* alpha, cudnnTensorDescriptor_t xDesc, void const* x,
        void const* beta, cudnnTensorDescriptor_t yDesc, void* y) {
        throw_if_failed(::cudnnLRNCrossChannelForward(super_type::get(), normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y));
      }

      void lrn_corss_channel_backward(
        cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
        void const* alpha,
        cudnnTensorDescriptor_t const yDesc, void const* y,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* beta,
        cudnnTensorDescriptor_t const dxDesc, void* dx) {
        throw_if_failed(::cudnnLRNCrossChannelBackward(super_type::get(), normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx));
      }

      void divisive_normalization_forward(
        cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode,
        void const* alpha, cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* means, void* temp, void* temp2,
        void const* beta, cudnnTensorDescriptor_t const yDesc, void* y) {
        throw_if_failed(::cudnnDivisiveNormalizationForward(super_type::get(), normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y));
      }

      void divisive_normalization_backward(
        cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode,
        void const* alpha, cudnnTensorDescriptor_t const xDesc, void const* x,
        void const* means,
        void const* dy, void* temp, void* temp2,
        void const* beta, cudnnTensorDescriptor_t const dxDesc, void* dx, void* dMeans) {
        throw_if_failed(::cudnnDivisiveNormalizationBackward(super_type::get(), normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dxDesc, dx, dMeans));
      }

      void batch_normalization_forward_inference(
        cudnnBatchNormMode_t mode,
        void const* alpha, void const* beta,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        cudnnTensorDescriptor_t const yDesc, void const* y,
        cudnnTensorDescriptor_t const bnScaleBiasMeanVarDesc, void const* bnScale, void const* bnBias,
        void const* estimatedMean, void const* estimatedInvVariance, double epsilon) {
        throw_if_failed(::cudnnBatchNormalizationForwardInference(super_type::get(), mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedInvVariance, epsilon));
      }

      void batch_normalization_forward_training(
        cudnnBatchNormMode_t mode,
        void const* alpha, void const* beta,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        cudnnTensorDescriptor_t const yDesc, void* y,
        cudnnTensorDescriptor_t const bnScaleBiasMeanVarDesc, void const* bnScale, void const* bnBias,
        double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningInvVariance,
        double epsilon,
        void* resultSaveMean, void* resultSaveInvVariance) {
        throw_if_failed(::cudnnBatchNormalizationForwardTraining(super_type::get(), mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningInvVariance, epsilon, resultSaveMean, resultRunningInvVariance));
      }

      void batch_normalization_backward(
        cudnnBatchNormMode_t mode,
        void const* alphaDataDiff, void const* betaDataDiff,
        void const* alphaParamDiff, void const* betaParamDiff,
        cudnnTensorDescriptor_t const xDesc, void const* x,
        cudnnTensorDescriptor_t const dyDesc, void const* dy,
        cudnnTensorDescriptor_t const dxDesc, void* dx,
        cudnnTensorDescriptor_t const bnScaleBiasDiffDesc, void const* bnScale,
        void* resultBnScaleDiff, void* resultBnBiasDiff,
        double epsilon,
        void const* savedMean, void const* savedInvVariance) {
        throw_if_failed(::cudnnBatchNormalizationBackward(super_type::get(), mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVariance));
      }

    public:
      inline static std::size_t version() {
        return ::cudnnGetVersion();
      }
    };

    class filter : std::shared_ptr<std::remove_pointer_t<cudnnFilterDescriptor_t>> {
      using super_type = std::shared_ptr<std::remove_pointer_t<cudnnFilterDescriptor_t>>;

    public:
      filter() {
        cudnnFilterDescriptor_t desc = nullptr;
        throw_if_failed(::cudnnCreateFilterDescriptor(&desc));
        super_type::reset(desc, [](cudnnFilterDescriptor_t desc) {
          ::cudnnDestroyFilterDescriptor(desc);
        });
      }

      void get(cudnnDataType_t* dataType, int* k, int* c, int* h, int* w) {
        throw_if_failed(::cudnnGetFilter4dDescriptor(super_type::get(), dataType, k, c, h, w));
      }

      void get(cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* k, int* c, int* h, int* w) {
        throw_if_failed(::cudnnGetFilter4dDescriptor_v4(super_type::get(), dataType, format, k, c, h, w));
      }

      void get(int nbDimsRequested, cudnnDataType_t* dataType, int* nbDims, int filterDimA[]) {
        throw_if_failed(::cudnnGetFilterNdDescriptor(super_type::get(), nbDimsRequested, dataType, nbDims, filterDimA));
      }

      void get(int nbDimsRequested, cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* nbDims, int filterDimA[]) {
        throw_if_failed(::cudnnGetFilterNdDescriptor_v4(super_type::get(), nbDimsRequested, dataType, format, nbDims, filterDimA));
      }

      void set(cudnnDataType_t dataType, int k, int c, int h, int w) {
        throw_if_failed(::cudnnSetFilter4dDescriptor(super_type::get(), dataType, k, c, h, w));
      }

      void set(cudnnDataType_t dataType, cudnnTensorFormat_t format, int k, int c, int h, int w) {
        throw_if_failed(::cudnnSetFilter4dDescriptor_v4(super_type::get(), dataType, format, k, c, h, w));
      }

      void set(int nbDimsRequested, cudnnDataType_t dataType, int nbDims, int filterDimA[]) {
        throw_if_failed(::cudnnSetFilterNdDescriptor(super_type::get(), dataType, nbDims, filterDimA));
      }

      void set(cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, int filterDimA[]) {
        throw_if_failed(::cudnnSetFilterNdDescriptor_v4(super_type::get(), dataType, format, nbDims, filterDimA));
      }
    };

    class convolution : std::shared_ptr<std::remove_pointer_t<cudnnConvolutionDescriptor_t>> {
      using super_type = std::shared_ptr<std::remove_pointer_t<cudnnConvolutionDescriptor_t>>;

    public:
      convolution() {
        cudnnConvolutionDescriptor_t desc = nullptr;
        throw_if_failed(::cudnnCreateConvolutionDescriptor(&desc));
        super_type::reset(desc, [](cudnnConvolutionDescriptor_t desc) {
          ::cudnnDestroyConvolutionDescriptor(desc);
        });
      }

      void get(int* pad_h, int* pad_w, int* u, int* v, int* upscalex, int* upscaley, cudnnConvolutionMode_t* mode) {
        throw_if_failed(::cudnnGetConvolution2dDescriptor(super_type::get(), pad_h, pad_w, u, v, upscalex, upscaley, mode));
      }

      void get(int arrayLengthRequested, int* arrayLength, int padA[], int filterStrideA[], int upscaleA[], cudnnConvolutionMode_t* mode, cudnnDataType_t* dataType) {
        throw_if_failed(::cudnnGetConvolutionNdDescriptor_v3(super_type::get(), arrayLengthRequested, arrayLength, padA, filterStrideA, upscaleA, mode, dataType));
      }

      void set(int pad_h, int pad_w, int u, int v, int upscalex, int upscaley, cudnnConvolutionMode_t mode) {
        throw_if_failed(::cudnnSetConvolution2dDescriptor(super_type::get(), pad_h, pad_w, u, v, upscalex, upscaley, mode));
      }

      void set(int arrayLength, int padA[], int filterStrideA[], int upscaleA[], cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
        throw_if_failed(::cudnnSetConvolutionNdDescriptor_v3(super_type::get(), arrayLength, padA, filterStrideA, upscaleA, mode, dataType));
      }

      void get_output_dim(cudnnTensorDescriptor_t const inputTensorDesc, cudnnFilterDescriptor_t const filterDesc, int* n, int* c, int* h, int* w) {
        throw_if_failed(::cudnnGetConvolution2dForwardOutputDim(super_type::get(), inputTensorDesc, filterDesc, n, c, h, w));
      }

      void get_output_dim(cudnnTensorDescriptor_t const inputTensorDesc, cudnnFilterDescriptor_t const filterDesc, int nbDims, int tensorOutputDimA[]) {
        throw_if_failed(::cudnnGetConvolutionNdForwardOutputDim(super_type::get(), inputTensorDesc, filterDesc, nbDims, tensorOutputDimA));
      }
    };

    class pooling : std::shared_ptr<std::remove_pointer_t<cudnnPoolingDescriptor_t>> {
      using super_type = std::shared_ptr<std::remove_pointer_t<cudnnPoolingDescriptor_t>>;

    public:
      pooling() {
        cudnnPoolingDescriptor_t desc = nullptr;
        throw_if_failed(::cudnnCreatePoolingDescriptor(&desc));
        super_type::reset(desc, [](cudnnPoolingDescriptor_t desc) {
          ::cudnnDestroyPoolingDescriptor(desc);
        });
      }

      void get(cudnnPoolingMode_t* mode, int* windowHeight, int* windowWidth,
        int* verticalPadding, int* horizontalPadding, int* verticalStride, int* horizontalStride) {
        throw_if_failed(::cudnnGetPooling2dDescriptor(super_type::get(), mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
      }

      void get(cudnnPoolingMode_t* mode, cudnnNanPropagation_t* maxpoolingNanOpt, int* windowHeight, int* windowWidth,
        int* verticalPadding, int* horizontalPadding, int* verticalStride, int* horizontalStride) {
        throw_if_failed(::cudnnGetPooling2dDescriptor_v4(super_type::get(), mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
      }

      void get(int nbDimsRequested, cudnnPoolingMode_t* mode, int* nbDims, int windowDimA[], int paddingA[], int strideA[]) {
        throw_if_failed(::cudnnGetPoolingNdDescriptor(super_type::get(), nbDimsRequested, mode, nbDims, windowDimA, paddingA, strideA));
      }

      void get(int nbDimsRequested, cudnnPoolingMode_t* mode, cudnnNanPropagation_t* maxpoolingNanOpt, int* nbDims, int windowDimA[], int paddingA[], int strideA[]) {
        throw_if_failed(::cudnnGetPoolingNdDescriptor_v4(super_type::get(), nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA));
      }

      void set(cudnnPoolingMode_t mode, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride) {
        throw_if_failed(::cudnnSetPooling2dDescriptor(super_type::get(), mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
      }

      void set(cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride) {
        throw_if_failed(::cudnnSetPooling2dDescriptor_v4(super_type::get(), mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));
      }

      void set(cudnnPoolingMode_t mode, int nbDims, int windowDimA[], int paddingA[], int strideA[]) {
        throw_if_failed(::cudnnSetPoolingNdDescriptor(super_type::get(), mode, nbDims, windowDimA, paddingA, strideA));
      }

      void set(cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int nbDims, int windowDimA[], int paddingA[], int strideA[]) {
        throw_if_failed(::cudnnSetPoolingNdDescriptor_v4(super_type::get(), mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA));
      }

      void get_output_dim(cudnnTensorDescriptor_t inputDesc, int* n, int* c, int* h, int* w) {
        throw_if_failed(::cudnnGetPooling2dForwardOutputDim(super_type::get(), inputDesc, n, c, h, w));
      }

      void get_output_dim(cudnnTensorDescriptor_t inputDesc, int nbDims, int outDimA[]) {
        throw_if_failed(::cudnnGetPoolingNdForwardOutputDim(super_type::get(), inputDesc, nbDims, outDimA));
      }
    };

    class activation : std::shared_ptr<std::remove_pointer_t<cudnnActivationDescriptor_t>> {
      using super_type = std::shared_ptr<std::remove_pointer_t<cudnnActivationDescriptor_t>>;

    public:
      activation() {
        cudnnActivationDescriptor_t desc = nullptr;
        throw_if_failed(::cudnnCreateActivationDescriptor(&desc));
        super_type::reset(desc, [](cudnnActivationDescriptor_t desc) {
          ::cudnnDestroyActivationDescriptor(desc);
        });
      }

      void get(cudnnActivationMode_t* mode, cudnnNanPropagation_t* reluNanOpt, double* reluCeiling) {
        throw_if_failed(::cudnnGetActivationDescriptor(super_type::get(), mode, reluNanOpt, reluCeiling));
      }

      void set(cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double reluCeiling) {
        throw_if_failed(::cudnnSetActivationDescriptor(super_type::get(), mode, reluNanOpt, reluCeiling));
      }
    };

    class local_response_normalization : std::shared_ptr<std::remove_pointer_t<cudnnLRNDescriptor_t>>{
      using super_type = std::shared_ptr<std::remove_pointer_t<cudnnLRNDescriptor_t>>;

    public:
      local_response_normalization() {
        cudnnLRNDescriptor_t desc = nullptr;
        throw_if_failed(::cudnnCreateLRNDescriptor(&desc));
        super_type::reset(desc, [](cudnnLRNDescriptor_t desc) {
          ::cudnnDestroyLRNDescriptor(desc);
        });
      }

      void get(unsigned* lrnN, double* lrnAlpha, double* lrnBeta, double* lrnK) {
        throw_if_failed(::cudnnGetLRNDescriptor(super_type::get(), lrnN, lrnAlpha, lrnBeta, lrnK));
      }

      void set(unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK) {
        throw_if_failed(::cudnnSetLRNDescriptor(super_type::get(), lrnN, lrnAlpha, lrnBeta, lrnK));
      }
    };
  }
}

#endif // REACTOR_MATH_DNN_HPP_INCLUDED
