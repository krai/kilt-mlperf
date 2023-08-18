//
// MIT License
//
// Copyright (c) 2023 Krai Ltd
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.POSSIBILITY OF SUCH DAMAGE.
//

#ifndef DEVICE_H
#define DEVICE_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "config/device_config.h"
#include "idatasource.h"
#include "idevice.h"
#include "imodel.h"
#include "memory_manager.h"
#include <NvInferRuntimeCommon.h> // Include necessary TensorRT runtime headers
#include <condition_variable>
#include <cuda_fp16.h>
#include <dlfcn.h> // Include for dynamic loading
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

using namespace KRAI;
using namespace nvinfer1;

template <typename Sample> class Device : public IDevice<Sample> {
public:
  void Construct(IModel *_model, IDataSource *_data_source, IConfig *_config,
                 int hw_id, std::vector<int> aff) {
    model = _model;
    data_source = _data_source;
    model_cfg = static_cast<IModelConfig *>(_config->model_cfg);

    TensorRTDeviceConfig *device_cfg =
        static_cast<TensorRTDeviceConfig *>(_config->device_cfg);
    const std::string modelPath = device_cfg->getModelRoot();
    numberOfStreams = device_cfg->getNumberOfStreams();
    maxSeqLen = device_cfg->getMaxSeqLen();
    maxBatchSize = device_cfg->getMaxBatchSize();
    seqLenTrace.resize(maxSeqLen);
    batchSizeTrace.resize(maxBatchSize);

    // read the model file into a buffer
    std::ifstream modelFile(modelPath, std::ios::binary | std::ios::ate);
    if (!modelFile.is_open()) {
      throw std::runtime_error("Failed to open model file: " + modelPath);
    }
    std::streamsize size = modelFile.tellg();
    modelFile.seekg(0, std::ios::beg);
    std::vector<char> modelData(size);
    if (!modelFile.read(modelData.data(), size)) {
      throw std::runtime_error("Failed to read model file");
    }

    cudaError_t resultCudaGetDeviceCount = cudaGetDeviceCount(&numberOfDevices);
    if (resultCudaGetDeviceCount != cudaSuccess) {
      throw std::runtime_error(
          "Cannot get the number of devices: " +
          std::string(cudaGetErrorString(resultCudaGetDeviceCount)));
    }
    std::cout << "Number of devices is " << numberOfDevices << std::endl;

    // Load the custom plugin library
    void *pluginHandle =
        dlopen(device_cfg->getPluginsPath().c_str(), RTLD_NOW);

    // Check if the library was loaded successfully
    if (!pluginHandle) {
      std::cerr << "Failed to load the plugin library: " << dlerror()
                << std::endl;
    }

    // Define function pointers for the plugin registration functions
    using RegisterPluginFunc =
        void (*)(nvinfer1::INetworkDefinition *, const void *);
    RegisterPluginFunc registerPlugin = reinterpret_cast<RegisterPluginFunc>(
        dlsym(pluginHandle, "registerPlugin"));

    // Register the custom plugin for engine deserialization
    if (registerPlugin) {
      registerPlugin(nullptr, nullptr);
    }

    // using smart pointers as NVidia recommends
    logger = std::make_unique<Logger>();
    runtime.reset(createInferRuntime(*logger));
    initLibNvInferPlugins(logger.get(), "");

    engines.resize(numberOfDevices * numberOfStreams);
    contexts.resize(numberOfDevices * numberOfStreams);
    for (int d = 0; d < numberOfDevices; d++) {
      cudaError_t resultCudaSetDevice = cudaSetDevice(d);
      if (resultCudaSetDevice != cudaSuccess) {
        throw std::runtime_error(
            "Cannot choose a device: " +
            std::string(cudaGetErrorString(resultCudaSetDevice)));
      }
      for (int s = 0; s < numberOfStreams; s++) {
        engines[d * numberOfStreams + s].reset(
            runtime->deserializeCudaEngine(modelData.data(), size));
        contexts[d * numberOfStreams + s].reset(
            engines[d * numberOfStreams + s]->createExecutionContext());
      }
    }

    dlclose(pluginHandle);

    // obtain the number of IO tensors in the engine
    const int numIOTensors = engines[0]->getNbIOTensors();

    // iterate over the bindings and populate vectors with input and output
    // tensor names
    inputTensorNames.clear();
    outputTensorNames.clear();
    for (int i = 0; i < numIOTensors; i++) {
      const char *tensorName = engines[0]->getIOTensorName(i);
      const TensorIOMode tensorIOMode = engines[0]->getTensorIOMode(tensorName);
      if (tensorIOMode == TensorIOMode::kINPUT) {
        // this is an input binding
        inputTensorNames.push_back(tensorName);
      } else if (tensorIOMode == TensorIOMode::kOUTPUT) {
        // this is an output binding
        outputTensorNames.push_back(tensorName);
      }
    }
    inputTensorNames.shrink_to_fit();
    outputTensorNames.shrink_to_fit();

    // compare the actual and the expected numbers of input and output tensors
    int actualInputCount = inputTensorNames.size();
    int actualOutputCount = outputTensorNames.size();
    int expectedInputCount = model_cfg->getInputCount();
    int expectedOutputCount = model_cfg->getOutputCount();
    if (expectedInputCount != actualInputCount ||
        expectedOutputCount != actualOutputCount) {
      throw std::runtime_error(
          "Error: Expected " + std::to_string(expectedInputCount) +
          " input tensors and " + std::to_string(expectedOutputCount) +
          " output tensors, but got " + std::to_string(actualInputCount) +
          " input tensors and " + std::to_string(actualOutputCount) +
          " output tensors from the inference engine.");
    }

    // check if an optimization profile is needed
    needOptimizationProfile = false;
    for (const auto &tensorName : inputTensorNames) {
      const Dims inputDims = engines[0]->getTensorShape(tensorName.c_str());
      for (int j = 0; j < inputDims.nbDims; j++) {
        if (inputDims.d[j] == -1) {
          needOptimizationProfile = true;
          break;
        }
      }
      if (needOptimizationProfile) {
        break;
      }
    }

    // create vectors of CUDA streams
    streams.resize(numberOfDevices);
    for (int d = 0; d < numberOfDevices; d++) {
      cudaError_t resultCudaSetDevice = cudaSetDevice(d);
      if (resultCudaSetDevice != cudaSuccess) {
        throw std::runtime_error(
            "Cannot choose a device: " +
            std::string(cudaGetErrorString(resultCudaSetDevice)));
      }
      streams[d].reserve(numberOfStreams);
      for (int i = 0; i < numberOfStreams; i++) {
        cudaStream_t *p_stream = new cudaStream_t;
        cudaError_t resultCudaStreamCreate = cudaStreamCreate(p_stream);
        if (resultCudaStreamCreate != cudaSuccess) {
          throw std::runtime_error(
              "Failed to create CUDA stream: " +
              std::string(cudaGetErrorString(resultCudaStreamCreate)));
        }
        streams[d].emplace_back(p_stream, cudaStreamDeleter());
      }
    }

    // calculate input and output buffers sizes
    inputDataSizes.resize(inputTensorNames.size());
    for (int i = 0; i < inputDataSizes.size(); i++) {
      DataType inputType =
          engines[0]->getTensorDataType(inputTensorNames[i].c_str());
      inputDataSizes[i] = determineDataSize(inputType);
    }
    outputDataSizes.resize(outputTensorNames.size());
    for (int i = 0; i < outputDataSizes.size(); i++) {
      DataType outputType =
          engines[0]->getTensorDataType(outputTensorNames[i].c_str());
      outputDataSizes[i] = determineDataSize(outputType);
    }
    std::vector<int> defaultInputSizes(model_cfg->getInputCount());
    for (int i = 0; i < defaultInputSizes.size(); i++) {
      defaultInputSizes[i] = model_cfg->getInputSize(i);
    }
    std::vector<int> defaultOutputSizes(model_cfg->getOutputCount());
    for (int i = 0; i < defaultOutputSizes.size(); i++) {
      defaultOutputSizes[i] = model_cfg->getOutputSize(i);
    }

    inputBufferSizes = calculateBufferSizes(inputTensorNames, defaultInputSizes,
                                            inputDataSizes);
    outputBufferSizes = calculateBufferSizes(
        outputTensorNames, defaultOutputSizes, outputDataSizes);

    // set up memory management strategy
    inputMemoryManagementStrategyName =
        device_cfg->getInputMemoryManagementStrategyName();
    outputMemoryManagementStrategyName =
        device_cfg->getOutputMemoryManagementStrategyName();
    memoryManagers.resize(numberOfDevices);
    for (int d = 0; d < numberOfDevices; d++) {
      cudaError_t resultCudaSetDevice = cudaSetDevice(d);
      if (resultCudaSetDevice != cudaSuccess) {
        throw std::runtime_error(
            "Cannot choose a device: " +
            std::string(cudaGetErrorString(resultCudaSetDevice)));
      }
      memoryManagers[d] = std::make_unique<MemoryManager>(
          inputMemoryManagementStrategyName, outputMemoryManagementStrategyName,
          numberOfStreams, inputBufferSizes, outputBufferSizes);
    }

    // allocate memory for input and output buffers
    for (int d = 0; d < numberOfDevices; d++) {
      cudaError_t resultCudaSetDevice = cudaSetDevice(d);
      if (resultCudaSetDevice != cudaSuccess) {
        throw std::runtime_error(
            "Cannot choose a device: " +
            std::string(cudaGetErrorString(resultCudaSetDevice)));
      }
      memoryManagers[d]->allocateMemoryBuffers();
    }

    deviceNumber = 0;

    // Create a vector of events to wait upon
    waitEvents.reserve(numberOfDevices * numberOfStreams);
    unsigned int flags = cudaEventDefault | cudaEventDisableTiming;
    for (int i = 0; i < numberOfDevices * numberOfStreams; i++) {
      waitEvents.emplace_back(
          std::unique_ptr<cudaEvent_t, CudaEventDeleter>(new cudaEvent_t));
    }
    for (int d = 0; d < numberOfDevices; d++) {
      cudaError_t resultCudaSetDevice = cudaSetDevice(d);
      if (resultCudaSetDevice != cudaSuccess) {
        throw std::runtime_error(
            "Cannot choose a device: " +
            std::string(cudaGetErrorString(resultCudaSetDevice)));
      }
      for (int i = 0; i < numberOfStreams; i++) {
        cudaError_t resultCudaEventCreate = cudaEventCreateWithFlags(
            waitEvents[d * numberOfStreams + i].get(), flags);
        if (resultCudaEventCreate != cudaSuccess) {
          throw std::runtime_error(
              "Failed to create CUDA event: " +
              std::string(cudaGetErrorString(resultCudaEventCreate)));
        }
      }
    }

    // Create threads for processing inputs
    std::unique_lock<std::mutex> queueLock(queueMutex);
    runningFlag = true;

    for (int i = 0; i < numberOfDevices * numberOfStreams; ++i) {
      threads.push_back(std::thread(&Device::processInputs, this, i));
    }

    queueLock.unlock();
  }

  virtual int Inference(std::vector<Sample> samples) {
    std::lock_guard<std::mutex> lock(queueMutex);
    inputQueue.push(samples);
    queueCondition.notify_one();
    return 0;
  }

  void processInputs(int threadId) {
    while (runningFlag) {

      int currentStreamNumber = threadId % numberOfStreams;
      int currentDeviceNumber =
          (threadId - currentStreamNumber) / numberOfStreams;

      std::vector<Sample> samples;
      {
        std::unique_lock<std::mutex> queueLock(queueMutex);
        queueCondition.wait(queueLock, [this] {
          return !this->inputQueue.empty() || !this->runningFlag;
        });

        if (!runningFlag) {
          break;
        }

        if (!inputQueue.empty()) {
          samples = inputQueue.front();
          inputQueue.pop();
        }
      }

      cudaError_t resultCudaSetDevice = cudaSetDevice(currentDeviceNumber);
      if (resultCudaSetDevice != cudaSuccess) {
        throw std::runtime_error(
            "Cannot choose a device: " +
            std::string(cudaGetErrorString(resultCudaSetDevice)));
      }

      // getting the stream from the pool
      cudaStream_t &stream =
          *(streams[currentDeviceNumber][currentStreamNumber]);

      // populate CPU input buffers from a datasource
      std::vector<void *> inputPointers =
          memoryManagers[currentDeviceNumber]->getRawCPUPointers(
              BufferType::INPUT, currentStreamNumber);
      model->configureWorkload(data_source, &samples, inputPointers);

      std::vector<size_t> dimensionValues = std::vector<size_t>();
#ifdef KILT_BENCHMARK_STANDALONE_BERT_SORTING
      if (needOptimizationProfile) {
        const int batchSize = samples.size();
        int seqLen = samples[0].second;
        seqLenTrace[seqLen - 1] += 1;
        batchSizeTrace[batchSize - 1] += 1;
        dimensionValues.push_back(batchSize * seqLen);
        dimensionValues.push_back(batchSize * seqLen);
        dimensionValues.push_back(batchSize + 1);
        dimensionValues.push_back(seqLen);
        int sl = 0;
        for (int s = 0; s < batchSize; s++) {
          sl += samples[s].second;
        }
        dimensionValues.push_back(2 * sl);
      }
#endif

      // sending the sequence of commands into the stream

      std::vector<size_t> actualInputSizes;
      if (needOptimizationProfile) {
        actualInputSizes.resize(inputTensorNames.size());
        for (int i = 0; i < actualInputSizes.size(); i++) {
          actualInputSizes[i] = dimensionValues[i] * inputDataSizes[i];
        }
      }
      // copy input data from CPU to GPU
      memoryManagers[currentDeviceNumber]->copyDataToDevice(
          currentStreamNumber, stream, actualInputSizes);

      if (needOptimizationProfile) {
        contexts[threadId]->setOptimizationProfileAsync(0, stream);
      }

      // specify buffers and shaped (if needed) for input and output tensors
      for (int j = 0; j < inputTensorNames.size(); j++) {
        const char *tensorName = inputTensorNames[j].c_str();
        if (needOptimizationProfile) {
          Dims newDims;
          newDims.nbDims = 1;
          newDims.d[0] = dimensionValues[j];
          bool resultSetTensorShape =
              contexts[threadId]->setInputShape(tensorName, newDims);
          if (!resultSetTensorShape) {
            throw std::runtime_error("Failed to set up an input shape");
          }
        }
        bool resultSetTensorAddress = contexts[threadId]->setTensorAddress(
            tensorName,
            memoryManagers[currentDeviceNumber]->getRawPointer(
                MemoryType::GPU, BufferType::INPUT, j, currentStreamNumber));
        if (!resultSetTensorAddress) {
          throw std::runtime_error("Failed to set tensor addresses for inputs");
        }
      }
      for (int j = 0; j < outputTensorNames.size(); j++) {
        bool resultSetTensorAddress = contexts[threadId]->setTensorAddress(
            outputTensorNames[j].c_str(),
            memoryManagers[currentDeviceNumber]->getRawPointer(
                MemoryType::GPU, BufferType::OUTPUT, j, currentStreamNumber));
        if (!resultSetTensorAddress) {
          throw std::runtime_error(
              "Failed to set tensor addresses for outputs");
        }
      }

      bool resultAllInputShapesSpecified =
          contexts[threadId]->allInputShapesSpecified();
      if (!resultAllInputShapesSpecified) {
        throw std::runtime_error("Not all input shapes are specified");
      }
      // run inference
      bool resultEnqueue = contexts[threadId]->enqueueV3(stream);
      if (!resultEnqueue) {
        throw std::runtime_error("Failed to run inference");
      }

      std::vector<size_t> actualOutputSizes;
      if (needOptimizationProfile) {
        actualOutputSizes.resize(outputTensorNames.size());
        for (int i = 0; i < actualOutputSizes.size(); i++) {
          actualOutputSizes[i] =
              dimensionValues[inputTensorNames.size() + i] * outputDataSizes[i];
        }
      }

      // copy output buffer from device
      memoryManagers[currentDeviceNumber]->copyDataFromDevice(
          currentStreamNumber, stream, actualOutputSizes);

      cudaEventRecord(*waitEvents[threadId], stream);
      cudaEventSynchronize(*waitEvents[threadId]);
      std::vector<void *> outputPointers =
          memoryManagers[currentDeviceNumber]->getRawCPUPointers(
              BufferType::OUTPUT, currentStreamNumber);
      model->postprocessResults(&samples, outputPointers);
    }
  }

  ~Device() {
    if (needOptimizationProfile) {
      std::cout << " Sequence lengths: ";
      for (int i = 0; i < seqLenTrace.size(); i++) {
        std::cout << seqLenTrace[i] << " ";
      }
      std::cout << std::endl;

      std::cout << " Batch sizes: ";
      for (int i = 0; i < batchSizeTrace.size(); i++) {
        std::cout << batchSizeTrace[i] << " ";
      }
      std::cout << std::endl;
    }
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      runningFlag = false;
      queueCondition.notify_all();
    }
    for (auto &thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

private:
  IModel *model;
  IDataSource *data_source;
  IModelConfig *model_cfg;

  int maxSeqLen;
  int maxBatchSize;
  std::vector<int> seqLenTrace;
  std::vector<int> batchSizeTrace;

  // TensorRT logger
  class Logger : public ILogger {
    void log(Severity severity, const char *msg) noexcept override {
      // suppress info-level messages
      if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
    }
  };
  std::unique_ptr<Logger> logger;

  int numberOfDevices;
  int deviceNumber;

  // Cuda and TensorRT objects
  std::unique_ptr<IRuntime> runtime;
  std::vector<std::unique_ptr<ICudaEngine>> engines;
  std::vector<std::unique_ptr<IExecutionContext>> contexts;

  // streams and their parameters
  struct cudaStreamDeleter {
    void operator()(cudaStream_t *stream) const {
      cudaError_t result = cudaStreamDestroy(*stream);
      if (result != cudaSuccess) {
        throw std::runtime_error("Failed to delete a Cuda stream: " +
                                 std::string(cudaGetErrorString(result)));
      }
    }
  };

  int numberOfStreams;
  std::vector<std::vector<std::unique_ptr<cudaStream_t, cudaStreamDeleter>>>
      streams;

  // a custom deleter for CUDA events
  struct CudaEventDeleter {
    void operator()(cudaEvent_t *event) const { cudaEventDestroy(*event); }
  };
  std::vector<std::unique_ptr<cudaEvent_t, CudaEventDeleter>> waitEvents;

  // vectors to hold potential input and output tensor names
  std::vector<std::string> inputTensorNames, outputTensorNames;
  std::vector<size_t> inputDataSizes, outputDataSizes;

  // vectors to hold sizes of buffers for input and output tensors
  std::vector<size_t> inputBufferSizes, outputBufferSizes;

  // memory management strategy
  std::string inputMemoryManagementStrategyName,
      outputMemoryManagementStrategyName;
  std::vector<std::unique_ptr<MemoryManager>> memoryManagers;

  bool needOptimizationProfile;

  // parameters for multithreading
  std::queue<std::vector<Sample>> inputQueue;
  std::mutex queueMutex;
  std::condition_variable queueCondition;
  std::vector<std::thread> threads;
  std::atomic<bool> runningFlag;

  // determines a datasize of a datatype used in a tensor
  size_t determineDataSize(DataType dataType) {
    size_t dataTypeSize = 0;
    switch (dataType) {
    case DataType::kFLOAT:
      dataTypeSize = sizeof(float);
      break;
    case DataType::kHALF:
      dataTypeSize = sizeof(__half);
      break;
    case DataType::kINT8:
      dataTypeSize = sizeof(int8_t);
      break;
    case DataType::kINT32:
      dataTypeSize = sizeof(int32_t);
      break;
    case DataType::kBOOL:
      dataTypeSize = sizeof(bool);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
    }
    return dataTypeSize;
  }

  // calculates buffer sizes for a vector of tensor names
  std::vector<size_t> calculateBufferSizes(std::vector<std::string> tensorNames,
                                           std::vector<int> defaultTensorSizes,
                                           std::vector<size_t> dataSizes) {
    std::vector<size_t> bufferSizes(tensorNames.size());
    for (int i = 0; i < tensorNames.size(); i++) {
      const char *tensorName = tensorNames[i].c_str();
      size_t tensorTypeSize = dataSizes[i];
      std::cout << "Found a tensor " << tensorName << " with dimensions ";
      Dims tensorDims = engines[0]->getTensorShape(tensorName);
      int tensorSize = 1;
      for (int j = 0; j < tensorDims.nbDims; j++) {
        std::cout << tensorDims.d[j] << ", ";
        if (tensorDims.d[j] != -1) {
          tensorSize *= tensorDims.d[j];
        } else {
          tensorSize *= defaultTensorSizes[i];
        }
      }
      std::cout << std::endl;
      bufferSizes[i] = tensorSize * tensorTypeSize;
      std::cout << "Tensor " << tensorName << " has the size of "
                << bufferSizes[i] << std::endl;
    }
    return bufferSizes;
  }
};

template <typename Sample>
IDevice<Sample> *createDevice(IModel *_model, IDataSource *_data_source,
                              IConfig *_config, int hw_id,
                              std::vector<int> aff) {
  Device<Sample> *d = new Device<Sample>();
  d->Construct(_model, _data_source, _config, hw_id, aff);
  return d;
}

#endif // DEVICE_H
