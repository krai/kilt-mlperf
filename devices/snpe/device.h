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

#include "config/device_config.h"
#include "idatasource.h"
#include "imodel.h"

#include "DiagLog/IDiagLog.hpp"
#include "DiagLog/Options.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/TensorShape.hpp"
#include "SNPE/ApplicationBufferMap.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"

#include <cassert>
#include <iostream>

#define LARGE_BUFFER 4000000
size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank,
                        size_t elementSize, size_t resizable_dim);
void createUserBuffer(
    zdl::DlSystem::UserBufferMap &userBufferMap,
    std::unordered_map<std::string, std::vector<uint8_t>> &applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
        &snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE> &snpe, const char *name);

using namespace KRAI;

template <typename Sample> class Device : public IDevice<Sample> {

public:
  void Construct(IModel *_model, IDataSource *_data_source, IConfig *_config,
                 int hw_id, std::vector<int> aff) {

    model = _model;
    data_source = _data_source;

    model_cfg = static_cast<IModelConfig *>(_config->model_cfg);

    device_cfg = static_cast<SnpeDeviceConfig *>(_config->device_cfg);

    static zdl::DlSystem::Version_t Version =
        zdl::SNPE::SNPEFactory::getLibraryVersion();

    const std::string outputDir = "./output/";

    std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl;

    // Get available runtimes
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(
            zdl::DlSystem::Runtime_t::DSP)) {
      std::cout << "DSP found" << std::endl;
    }
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(
            zdl::DlSystem::Runtime_t::AIP_FIXED8_TF)) {
      std::cout << "AIP found" << std::endl;
    }
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(
            zdl::DlSystem::Runtime_t::GPU)) {
      std::cout << "GPU found" << std::endl;
    }
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(
            zdl::DlSystem::Runtime_t::CPU)) {
      std::cout << "CPU found" << std::endl;
    }

    // Attempt to open dlc file from path
    std::string container_path = device_cfg->getModelRoot();
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(container_path);

    if (container == nullptr) {
      std::cerr << "Error while opening the container file." << std::endl;
    } else {
      std::cout << "Successfully loaded " << container_path << std::endl;
    }

    // Set network builder options
    std::string backend_type = device_cfg->getBackendType();
    std::cout << "Using backend: " << backend_type << std::endl;
    zdl::DlSystem::RuntimeList runtimeList;
    runtimeList.add(
        zdl::DlSystem::RuntimeList::stringToRuntime(backend_type.c_str()));

    // Set output names
    zdl::DlSystem::StringList output_names;

    bool useUserSuppliedBuffers = true;
    bool useInitCache = true;

    zdl::DlSystem::TensorShapeMap inputShapeMap;
    size_t batch_size = model_cfg->getBatchSize();
    std::cout << "Setting batch size to " << batch_size << std::endl;
    inputShapeMap.add("input_tensor_1:0", {batch_size, 224, 224, 3});

    std::string performance_profile = device_cfg->getPerformanceProfile();
    std::cout << "Setting performance profile to " << performance_profile
              << std::endl;
    zdl::DlSystem::PerformanceProfile_t performanceProfile;
    if (performance_profile == "default")
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::DEFAULT;
    else if (performance_profile == "balanced")
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::BALANCED;
    else if (performance_profile == "power_saver")
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::POWER_SAVER;
    else if (performance_profile == "system_settings")
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::SYSTEM_SETTINGS;
    else if (performance_profile == "sustained_high_performance")
      performanceProfile =
          zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE;
    else if (performance_profile == "burst")
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::BURST;
    else if (performance_profile == "low_power_saver")
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::LOW_POWER_SAVER;
    else if (performance_profile == "high_power_saver")
      performanceProfile =
          zdl::DlSystem::PerformanceProfile_t::HIGH_POWER_SAVER;
    else if (performance_profile == "low_balanced")
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::LOW_BALANCED;
    else {
      std::cerr << performance_profile << "is not a valid performance profile,"
                << "setting to DEFAULT" << std::endl;
      performanceProfile = zdl::DlSystem::PerformanceProfile_t::DEFAULT;
    }

    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setPerformanceProfile(performanceProfile)
               .setRuntimeProcessorOrder(runtimeList)
               .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
               .setInitCacheMode(useInitCache)
               .setInputDimensions(inputShapeMap)
               .build();

    if (snpe == nullptr) {
      std::cerr << "Error while building SNPE object." << std::endl;
    }

    // set up cache of container
    if (useInitCache) {
      if (container->save(container_path)) {
        std::cout << "Saved container into archive successfully" << std::endl;
      } else {
        std::cerr << "Failed to save container into archive" << std::endl;
      }
    }

    // Get network information
    // getting input dimensions
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    size_t num_input_dimensions = tensorShape.rank();
    std::cout << "Network input dimensions: ";
    for (int i = 0; i < num_input_dimensions; i++)
      std::cout << tensorShape.getDimensions()[i] << " ";
    std::cout << std::endl;

    // getting input names
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt)
      throw std::runtime_error("Error obtaining input tensor names");
    inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    std::cout << "Input names:";
    for (const char *name : inputNames)
      std::cout << " " << name;
    std::cout << std::endl;

    // getting output names
    const auto &outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt)
      throw std::runtime_error("Error obtaining output tensor names");
    outputNames = *outputNamesOpt;
    assert(outputNames.size() > 0);

    std::cout << "Output names:";
    for (const char *name : outputNames)
      std::cout << " " << name;
    std::cout << std::endl;

    // Creating user buffers
    for (const char *name : inputNames)
      createUserBuffer(inputMap, applicationInputBuffers,
                       snpeUserBackedInputBuffers, snpe, name);
    for (const char *name : outputNames)
      createUserBuffer(outputMap, applicationOutputBuffers,
                       snpeUserBackedOutputBuffers, snpe, name);
  }

  virtual int Inference(std::vector<Sample> samples) {

    // get application input and output buffers
    for (std::unordered_map<std::string, std::vector<uint8_t>>::iterator it =
             applicationInputBuffers.begin();
         it != applicationInputBuffers.end(); ++it) {
      buffers_in.push_back(it->second.data());
    }
    for (std::unordered_map<std::string, std::vector<uint8_t>>::iterator it =
             applicationOutputBuffers.begin();
         it != applicationOutputBuffers.end(); ++it) {
      buffers_out.push_back(it->second.data());
    }

    // populate device input buffers from datasource
    model->configureWorkload(data_source, &samples, buffers_in);

    // do device specific inference here
    if (!snpe->execute(inputMap, outputMap)) {
      std::cerr << "Error while executing the network";
    }

    // pass device output buffers to model specific post processing
    model->postprocessResults(&samples, buffers_out);

    return 0;
  }

  ~Device() {
    zdl::SNPE::SNPEFactory::terminateLogging();
    snpe.reset();
  }

private:
  SnpeDeviceConfig *device_cfg;
  IModelConfig *model_cfg;

  // activations, set, input buffers
  std::vector<void *> buffers_in;
  // activation, set, output buffers
  std::vector<void *> buffers_out;

  IModel *model;
  IDataSource *data_source;

  std::unique_ptr<zdl::SNPE::SNPE> snpe;

  // buffer maps: "handles" for application buffers, are later fed to the
  // network for execution
  zdl::DlSystem::UserBufferMap inputMap, outputMap;
  // user-backed buffers: what SNPE interacts with, a layer of buffers on top of
  // application buffers
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
      snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
  // application buffers: what you interact with, load/read data from
  std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers,
      applicationOutputBuffers;

  zdl::DlSystem::StringList inputNames;
  zdl::DlSystem::StringList outputNames;

  void createUserBuffer(
      zdl::DlSystem::UserBufferMap &userBufferMap,
      std::unordered_map<std::string, std::vector<uint8_t>> &applicationBuffers,
      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
          &snpeUserBackedBuffers,
      std::unique_ptr<zdl::SNPE::SNPE> &snpe, const char *name) {
    // get attributes of buffer by name
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt)
      throw std::runtime_error(
          std::string("Error obtaining attributes for tensor ") + name);
    // calculate the size of buffer required by the input tensor
    const zdl::DlSystem::TensorShape &bufferShape =
        (*bufferAttributesOpt)->getDims();
    // Calculate the stride based on buffer strides, assuming tightly packed.
    // Note: Strides = Number of bytes to advance to the next element in each
    // dimension. For example, if a float tensor of dimension 2x4x3 is tightly
    // packed in a buffer of 96 bytes, then the strides would be (48,12,4) Note:
    // Buffer stride is usually known and does not need to be calculated. const
    // size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
    const size_t bufferElementSize = sizeof(float);
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = bufferElementSize;
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
      stride *= bufferShape[i];
      strides[i - 1] = stride;
    }
    // resizableDim = maximum size resizable dimensions can grow to
    // this allows us to allocate enough space if dimensions aren't known
    int resizableDim = 1024;
    size_t bufSize =
        calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(),
                         bufferElementSize, resizableDim);
    // set the buffer encoding type
    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));
    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory &ubFactory =
        zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(
        ubFactory.createUserBuffer(applicationBuffers.at(name).data(), bufSize,
                                   strides, &userBufferEncodingFloat));
    // add the user-backed buffer to the inputMap, which is later on fed to the
    // network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
  }

  size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank,
                          size_t elementSize, size_t resizable_dim) {
    if (rank == 0)
      return 0;
    size_t size = elementSize;
    while (rank--) {
      (*dims == 0) ? size *= resizable_dim : size *= *dims;
      dims++;
    }
    return size;
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
