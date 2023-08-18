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
#include <assert.h>
#include <fstream>
#include <onnxruntime_cxx_api.h>

#define LARGE_BUFFER 4000000

using namespace KRAI;

template <typename Sample> class Device : public IDevice<Sample> {

public:
  void Construct(IModel *_model, IDataSource *_data_source, IConfig *_config,
                 int hw_id, std::vector<int> aff) {

    model = _model;
    data_source = _data_source;

    model_cfg = static_cast<IModelConfig *>(_config->model_cfg);

    device_cfg = static_cast<OnnxDeviceConfig *>(_config->device_cfg);

    // load model from config path
    std::string model_path = device_cfg->getModelRoot();
    std::ifstream f(model_path.c_str());
    if (f.good())
      std::cout << "Loading model at " << model_path << std::endl;
    else
      std::cerr << "Model not found at " << model_path << std::endl;

    Ort::SessionOptions session_options;

    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "ort_resnet50");

    auto providers = Ort::GetAvailableProviders();
    std::cout << "Available providers:" << std::endl;
    for (auto provider : providers) {
      std::cout << " - " << provider << std::endl;
    }

    std::string backend_type = device_cfg->getBackendType();
    std::cout << "Using backend: " << backend_type << std::endl;

    if (backend_type == "gpu") {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    session = new Ort::Session{env, model_path.c_str(), session_options};
    if (!session) {
      std::cerr << "Failed to create Onnxrt session" << std::endl;
    } else {
      std::cout << "Successfully created Onnxrt session" << std::endl;
    }

    // get num of input and output nodes
    numInputNodes = session->GetInputCount();
    assert(numInputNodes == model_cfg->getInputCount());
    numOutputNodes = session->GetOutputCount();
    assert(numOutputNodes == model_cfg->getOutputCount());

    // get input info from ONNX file
    for (int i = 0; i < numInputNodes; i++) {

      // get type
      Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
      auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
      ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
      input_types.push_back(inputType);

      // get shape
      std::vector<int> input_dims_ = model_cfg->getInputDimensions(i);
      std::vector<int64_t> input_dims;
      for (int i : input_dims_)
        input_dims.push_back(static_cast<int64_t>(i));
      input_shapes.push_back(input_dims);

      // get name
      Ort::AllocatedStringPtr inputNamePtr =
          session->GetInputNameAllocated(i, allocator);
      std::string inputName = inputNamePtr.get();
      char *inputNameChar = strdup(inputName.data());
      input_names.push_back(inputNameChar);
    }

    // get output info from ONNX file
    for (int i = 0; i < numOutputNodes; i++) {

      // get type
      Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
      auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
      ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
      output_types.push_back(outputType);

      // get shape
      std::vector<int> output_dims_ = model_cfg->getOutputDimensions(i);
      std::vector<int64_t> output_dims;
      for (int i : output_dims_)
        output_dims.push_back(static_cast<int64_t>(i));
      output_shapes.push_back(output_dims);

      // get name
      Ort::AllocatedStringPtr outputNamePtr =
          session->GetOutputNameAllocated(i, allocator);
      std::string outputName = outputNamePtr.get();
      char *outputNameChar = strdup(outputName.data());
      output_names.push_back(outputNameChar);
    }

    // create dummy input buffers for device
    for (int i = 0; i < numInputNodes; ++i) {
      uint8_t *raw_input = new uint8_t[model_cfg->getInputByteSize(i)];
      buffers_in.push_back(raw_input);
    }

    // create dummy output buffers for device
    for (int i = 0; i < numOutputNodes; ++i) {
      uint8_t *raw_output = new uint8_t[model_cfg->getOutputByteSize(i)];
      buffers_out.push_back(raw_output);
    }
  }

  virtual int Inference(std::vector<Sample> samples) {

    // populate device input buffers from datasource
    model->configureWorkload(data_source, &samples, buffers_in);

    // do device specific inference here
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    for (int xx = 0; xx < numInputNodes; ++xx) {
      std::vector<int64_t> dims = input_shapes.at(xx);
      inputs.push_back(Ort::Value::CreateTensor(
          memory_info, buffers_in.at(xx), model_cfg->getInputByteSize(xx),
          dims.data(), dims.size(), input_types.at(xx)));
    }

    for (int xx = 0; xx < numOutputNodes; ++xx) {
      std::vector<int64_t> dims = output_shapes.at(xx);
      outputs.push_back(Ort::Value::CreateTensor(
          memory_info, buffers_out.at(xx), model_cfg->getOutputByteSize(xx),
          dims.data(), dims.size(), output_types.at(xx)));
    }

    Ort::RunOptions run_options;
    session->Run(run_options, input_names.data(), inputs.data(), numInputNodes,
                 output_names.data(), outputs.data(), numOutputNodes);

    // pass device output buffers to model specific post processing
    model->postprocessResults(&samples, buffers_out);

    return 0;
  }

  ~Device() { delete session; }

private:
  OnnxDeviceConfig *device_cfg;
  IModelConfig *model_cfg;

  // activations, set, input buffers
  std::vector<void *> buffers_in;

  // activation, set, output buffers
  std::vector<void *> buffers_out;

  IModel *model;
  IDataSource *data_source;

  std::vector<Ort::Value> inputs;
  std::vector<Ort::Value> outputs;

  size_t numInputNodes;
  size_t numOutputNodes;

  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<ONNXTensorElementDataType> output_types;

  std::vector<const char *> input_names;
  std::vector<const char *> output_names;

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;

  Ort::Env env;
  Ort::Session *session;

  Ort::AllocatorWithDefaultOptions allocator;
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
