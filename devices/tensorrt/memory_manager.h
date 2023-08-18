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

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "NvInfer.h"
#include <cuda_fp16.h>
#include <memory>

enum class StrategyType { PAGED_MEMORY, PINNED_MEMORY, MANAGED_MEMORY };
enum class MemoryType { CPU_PAGED, CPU_PINNED, GPU };
enum class BufferType { INPUT, OUTPUT };

class MemoryManager {
public:
  struct BufferDeleter {
    MemoryType memoryType;
    BufferDeleter(MemoryType memType) : memoryType(memType) {}

    void operator()(void *buffer) const {
      if (memoryType == MemoryType::CPU_PAGED) {
        std::free(buffer);
      } else if (memoryType == MemoryType::CPU_PINNED) {
        cudaError_t result = cudaFreeHost(buffer);
        if (result != cudaSuccess) {
          throw std::runtime_error("Failed to delete a Cuda buffer: " +
                                   std::string(cudaGetErrorString(result)));
        }
      } else if (memoryType == MemoryType::GPU) {
        cudaError_t result = cudaFree(buffer);
        if (result != cudaSuccess) {
          throw std::runtime_error("Failed to delete a Cuda buffer: " +
                                   std::string(cudaGetErrorString(result)));
        }
      } else {
        auto *vec = static_cast<std::vector<char> *>(buffer);
        delete vec;
      }
    }
  };
  using BufferStorage =
      std::vector<std::vector<std::unique_ptr<void, BufferDeleter>>>;

  MemoryManager(std::string inputStrategyName, std::string outputStrategyName,
                int numberOfStreams, std::vector<size_t> inputBufferSizes,
                std::vector<size_t> outputBufferSizes)
      : numberOfStreams(numberOfStreams), inputBufferSizes(inputBufferSizes),
        outputBufferSizes(outputBufferSizes) {
    inputStrategy = getStrategyType(inputStrategyName);
    outputStrategy = getStrategyType(outputStrategyName);

    if (inputStrategy == StrategyType::PAGED_MEMORY ||
        inputStrategy == StrategyType::PINNED_MEMORY) {
      inputBuffersCPU.resize(numberOfStreams);
      inputBuffersGPU.resize(numberOfStreams);
    } else if (inputStrategy == StrategyType::MANAGED_MEMORY) {
      inputBuffersShared.resize(numberOfStreams);
    }
    if (outputStrategy == StrategyType::PAGED_MEMORY ||
        outputStrategy == StrategyType::PINNED_MEMORY) {
      outputBuffersCPU.resize(numberOfStreams);
      outputBuffersGPU.resize(numberOfStreams);
    } else if (outputStrategy == StrategyType::MANAGED_MEMORY) {
      outputBuffersShared.resize(numberOfStreams);
    }
  }

  void allocateMemoryBuffers() {
    BufferDeleter cpuPagedBufferDeleter(MemoryType::CPU_PAGED);
    BufferDeleter cpuPinnedBufferDeleter(MemoryType::CPU_PINNED);
    BufferDeleter cudaBufferDeleter(MemoryType::GPU);
    if (inputStrategy == StrategyType::PAGED_MEMORY) {
      for (int i = 0; i < numberOfStreams; i++) {
        for (int j = 0; j < inputBufferSizes.size(); j++) {
          void *cpuBuffer = allocateCpuPagedMemory(inputBufferSizes[j]);
          inputBuffersCPU[i].emplace_back(cpuBuffer, cpuPagedBufferDeleter);
          void *gpuBuffer = allocateGpuMemory(inputBufferSizes[j]);
          inputBuffersGPU[i].emplace_back(gpuBuffer, cudaBufferDeleter);
        }
      }
    } else if (inputStrategy == StrategyType::PINNED_MEMORY) {
      for (int i = 0; i < numberOfStreams; i++) {
        for (int j = 0; j < inputBufferSizes.size(); j++) {
          void *cpuBuffer = allocateCpuPinnedMemory(inputBufferSizes[j]);
          inputBuffersCPU[i].emplace_back(cpuBuffer, cpuPinnedBufferDeleter);
          void *gpuBuffer = allocateGpuMemory(inputBufferSizes[j]);
          inputBuffersGPU[i].emplace_back(gpuBuffer, cudaBufferDeleter);
        }
      }
    } else if (inputStrategy == StrategyType::MANAGED_MEMORY) {
      for (int i = 0; i < numberOfStreams; i++) {
        for (int j = 0; j < inputBufferSizes.size(); j++) {
          void *sharedBuffer = allocateManagedMemory(inputBufferSizes[j]);
          inputBuffersShared[i].emplace_back(sharedBuffer, cudaBufferDeleter);
        }
      }
    }
    if (outputStrategy == StrategyType::PAGED_MEMORY) {
      for (int i = 0; i < numberOfStreams; i++) {
        for (int j = 0; j < outputBufferSizes.size(); j++) {
          void *cpuBuffer = allocateCpuPagedMemory(outputBufferSizes[j]);
          outputBuffersCPU[i].emplace_back(cpuBuffer, cpuPagedBufferDeleter);
          void *gpuBuffer = allocateGpuMemory(outputBufferSizes[j]);
          outputBuffersGPU[i].emplace_back(gpuBuffer, cudaBufferDeleter);
        }
      }
    } else if (outputStrategy == StrategyType::PINNED_MEMORY) {
      for (int i = 0; i < numberOfStreams; i++) {
        for (int j = 0; j < outputBufferSizes.size(); j++) {
          void *cpuBuffer = allocateCpuPinnedMemory(outputBufferSizes[j]);
          outputBuffersCPU[i].emplace_back(cpuBuffer, cpuPinnedBufferDeleter);
          void *gpuBuffer = allocateGpuMemory(outputBufferSizes[j]);
          outputBuffersGPU[i].emplace_back(gpuBuffer, cudaBufferDeleter);
        }
      }
    } else if (outputStrategy == StrategyType::MANAGED_MEMORY) {
      for (int i = 0; i < numberOfStreams; i++) {
        for (int j = 0; j < outputBufferSizes.size(); j++) {
          void *sharedBuffer = allocateManagedMemory(outputBufferSizes[j]);
          outputBuffersShared[i].emplace_back(sharedBuffer, cudaBufferDeleter);
        }
      }
    }
  }

  void copyDataToDevice(int streamNumber, cudaStream_t &stream,
                        const std::vector<size_t> &customInputBufferSizes =
                            std::vector<size_t>()) {
    const std::vector<size_t> &bufferSizes = customInputBufferSizes.empty()
                                                 ? inputBufferSizes
                                                 : customInputBufferSizes;
    if (inputStrategy == StrategyType::PAGED_MEMORY ||
        inputStrategy == StrategyType::PINNED_MEMORY) {
      for (int i = 0; i < inputBuffersGPU[streamNumber].size(); i++) {
        cudaError_t resultCudaMemcpyAsync =
            cudaMemcpyAsync(inputBuffersGPU[streamNumber][i].get(),
                            inputBuffersCPU[streamNumber][i].get(),
                            bufferSizes[i], cudaMemcpyHostToDevice, stream);
        if (resultCudaMemcpyAsync != cudaSuccess) {
          throw std::runtime_error(
              "Failed to copy data to GPU: " +
              std::string(cudaGetErrorString(resultCudaMemcpyAsync)));
        }
      }
    } else if (inputStrategy == StrategyType::MANAGED_MEMORY) {
      for (int i = 0; i < inputBuffersShared[streamNumber].size(); i++) {
        cudaError_t resultCudaStreamAttachMemAsync = cudaStreamAttachMemAsync(
            stream, inputBuffersShared[streamNumber][i].get(), 0,
            cudaMemAttachSingle);
        if (resultCudaStreamAttachMemAsync != cudaSuccess) {
          throw std::runtime_error(
              "Failed to attach memory to the stream: " +
              std::string(cudaGetErrorString(resultCudaStreamAttachMemAsync)));
        }
      }
    }
    if (outputStrategy == StrategyType::MANAGED_MEMORY) {
      for (int i = 0; i < outputBuffersShared[streamNumber].size(); i++) {
        cudaError_t resultCudaStreamAttachMemAsync = cudaStreamAttachMemAsync(
            stream, outputBuffersShared[streamNumber][i].get(), 0,
            cudaMemAttachSingle);
        if (resultCudaStreamAttachMemAsync != cudaSuccess) {
          throw std::runtime_error(
              "Failed to attach memory to the stream: " +
              std::string(cudaGetErrorString(resultCudaStreamAttachMemAsync)));
        }
      }
    }
  }

  void copyDataFromDevice(int streamNumber, cudaStream_t &stream,
                          const std::vector<size_t> &customOutputBufferSizes =
                              std::vector<size_t>()) {
    const std::vector<size_t> &bufferSizes = customOutputBufferSizes.empty()
                                                 ? outputBufferSizes
                                                 : customOutputBufferSizes;
    if (outputStrategy == StrategyType::PAGED_MEMORY ||
        outputStrategy == StrategyType::PINNED_MEMORY) {
      for (int i = 0; i < outputBuffersGPU[streamNumber].size(); i++) {
        cudaError_t resultCudaMemcpyAsync =
            cudaMemcpyAsync(outputBuffersCPU[streamNumber][i].get(),
                            outputBuffersGPU[streamNumber][i].get(),
                            bufferSizes[i], cudaMemcpyDeviceToHost, stream);
        if (resultCudaMemcpyAsync != cudaSuccess) {
          throw std::runtime_error(
              "Failed to copy data from GPU: " +
              std::string(cudaGetErrorString(resultCudaMemcpyAsync)));
        }
      }
    } else if (outputStrategy == StrategyType::MANAGED_MEMORY) {
      for (int i = 0; i < outputBuffersShared[streamNumber].size(); i++) {
        cudaError_t resultCudaStreamAttachMemAsync = cudaStreamAttachMemAsync(
            stream, outputBuffersShared[streamNumber][i].get(), 0,
            cudaMemAttachHost);
        if (resultCudaStreamAttachMemAsync != cudaSuccess) {
          throw std::runtime_error(
              "Failed to attach memory to the stream: " +
              std::string(cudaGetErrorString(resultCudaStreamAttachMemAsync)));
        }
      }
    }
    if (inputStrategy == StrategyType::MANAGED_MEMORY) {
      for (int i = 0; i < inputBuffersShared[streamNumber].size(); i++) {
        cudaError_t resultCudaStreamAttachMemAsync = cudaStreamAttachMemAsync(
            stream, inputBuffersShared[streamNumber][i].get(), 0,
            cudaMemAttachHost);
        if (resultCudaStreamAttachMemAsync != cudaSuccess) {
          throw std::runtime_error(
              "Failed to attach memory to the stream: " +
              std::string(cudaGetErrorString(resultCudaStreamAttachMemAsync)));
        }
      }
    }
    if (inputStrategy == StrategyType::MANAGED_MEMORY ||
        outputStrategy == StrategyType::MANAGED_MEMORY) {
      cudaError_t resultCudaDeviceSynchronize = cudaDeviceSynchronize();
      if (resultCudaDeviceSynchronize != cudaSuccess) {
        throw std::runtime_error(
            "Failed to synchronize the device: " +
            std::string(cudaGetErrorString(resultCudaDeviceSynchronize)));
      }
    }
  }

  std::vector<void *> getRawCPUPointers(BufferType bufferType,
                                        int streamNumber) {
    MemoryType memoryType;
    StrategyType strategyType;
    if (bufferType == BufferType::INPUT)
      strategyType = inputStrategy;
    else if (bufferType == BufferType::OUTPUT)
      strategyType = outputStrategy;
    else
      throw std::runtime_error("Cannot recognize a buffer type");

    if (strategyType == StrategyType::PAGED_MEMORY)
      memoryType = MemoryType::CPU_PAGED;
    else if (strategyType == StrategyType::PINNED_MEMORY)
      memoryType = MemoryType::CPU_PINNED;
    else if (strategyType == StrategyType::MANAGED_MEMORY)
      memoryType = MemoryType::GPU;
    else
      throw std::runtime_error("Cannot recognize a memory management strategy");

    BufferStorage *pStorage = getBufferPointer(memoryType, bufferType);
    int pointersSize = (*pStorage)[streamNumber].size();
    std::vector<void *> pointers(pointersSize);
    for (int i = 0; i < pointersSize; i++) {
      pointers[i] = (*pStorage)[streamNumber][i].get();
    }
    return pointers;
  }

  void *getRawPointer(MemoryType memoryType, BufferType bufferType,
                      int bufferNumber, int streamNumber) {
    BufferStorage *pStorage = getBufferPointer(memoryType, bufferType);
    return ((*pStorage)[streamNumber][bufferNumber]).get();
  }

private:
  StrategyType inputStrategy, outputStrategy;
  const int numberOfStreams;
  const std::vector<size_t> inputBufferSizes, outputBufferSizes;

  BufferStorage inputBuffersCPU, outputBuffersCPU;
  BufferStorage inputBuffersGPU, outputBuffersGPU;
  BufferStorage inputBuffersShared, outputBuffersShared;

  void *allocateCpuPagedMemory(size_t bufferSize) {
    void *buffer = std::aligned_alloc(256, bufferSize);
    return buffer;
  }

  void *allocateGpuMemory(size_t bufferSize) {
    void *buffer = nullptr;
    cudaError_t resultCudaMalloc = cudaMalloc(&buffer, bufferSize);
    if (resultCudaMalloc != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate memory on GPU: " +
          std::string(cudaGetErrorString(resultCudaMalloc)));
    }
    return buffer;
  }

  void *allocateManagedMemory(size_t bufferSize) {
    void *buffer = nullptr;
    cudaError_t resultCudaMalloc =
        cudaMallocManaged(&buffer, bufferSize, cudaMemAttachHost);
    if (resultCudaMalloc != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate managed memory: " +
          std::string(cudaGetErrorString(resultCudaMalloc)));
    }
    return buffer;
  }

  void *allocateCpuPinnedMemory(size_t bufferSize) {
    void *buffer = nullptr;
    cudaError_t resultCudaMalloc = cudaMallocHost(&buffer, bufferSize);
    if (resultCudaMalloc != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate pinned memory on CPU: " +
          std::string(cudaGetErrorString(resultCudaMalloc)));
    }
    return buffer;
  }

  BufferStorage *getBufferPointer(MemoryType memoryType,
                                  BufferType bufferType) {
    BufferStorage *pStorage = nullptr;
    if ((inputStrategy == StrategyType::PAGED_MEMORY ||
         inputStrategy == StrategyType::PINNED_MEMORY) &&
        (memoryType == MemoryType::CPU_PAGED ||
         memoryType == MemoryType::CPU_PINNED) &&
        (bufferType == BufferType::INPUT))
      pStorage = &inputBuffersCPU;
    if ((outputStrategy == StrategyType::PAGED_MEMORY ||
         outputStrategy == StrategyType::PINNED_MEMORY) &&
        (memoryType == MemoryType::CPU_PAGED ||
         memoryType == MemoryType::CPU_PINNED) &&
        (bufferType == BufferType::OUTPUT))
      pStorage = &outputBuffersCPU;
    if ((inputStrategy == StrategyType::PAGED_MEMORY ||
         inputStrategy == StrategyType::PINNED_MEMORY) &&
        (memoryType == MemoryType::GPU) && (bufferType == BufferType::INPUT))
      pStorage = &inputBuffersGPU;
    if ((outputStrategy == StrategyType::PAGED_MEMORY ||
         outputStrategy == StrategyType::PINNED_MEMORY) &&
        (memoryType == MemoryType::GPU) && (bufferType == BufferType::OUTPUT))
      pStorage = &outputBuffersGPU;
    if ((inputStrategy == StrategyType::MANAGED_MEMORY) &&
        (bufferType == BufferType::INPUT))
      pStorage = &inputBuffersShared;
    if ((outputStrategy == StrategyType::MANAGED_MEMORY) &&
        (bufferType == BufferType::OUTPUT))
      pStorage = &outputBuffersShared;
    if (!pStorage)
      throw std::runtime_error("Cannot acquire a buffer pointer");
    return pStorage;
  }

  const StrategyType getStrategyType(std::string strategyName) const {
    StrategyType strategyType;
    if (strategyName == "PAGED_MEMORY")
      strategyType = StrategyType::PAGED_MEMORY;
    else if (strategyName == "PINNED_MEMORY")
      strategyType = StrategyType::PINNED_MEMORY;
    else if (strategyName == "MANAGED_MEMORY")
      strategyType = StrategyType::MANAGED_MEMORY;
    else
      strategyType = StrategyType::PAGED_MEMORY;
    return strategyType;
  }
};

#endif // MEMORY_MANAGER_H