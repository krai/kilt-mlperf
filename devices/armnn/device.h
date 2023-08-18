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

#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Tensor.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"

#include <iostream>

using namespace KRAI;

template <typename Sample> class Device : public IDevice<Sample> {

public:
  void Construct(IModel *_model, IDataSource *_data_source, IConfig *_config,
                 int hw_id, std::vector<int> aff) {

    model = _model;
    data_source = _data_source;

    model_cfg = static_cast<IModelConfig *>(_config->model_cfg);

    // load model from config path
    std::string model_path = "";
    armnn::INetworkPtr network =
        parser->CreateNetworkFromBinaryFile(model_path.c_str());
    if (!network)
      throw "Failed to load graph from file";

    // TODO: Get these from config
    bool use_neon = false;
    bool use_opencl = false;
    // Optimize the network for a specific runtime compute device, e.g. CpuAcc,
    // GpuAcc
    // std::vector<armnn::BackendId> optOptions = {armnn::Compute::CpuAcc,
    // armnn::Compute::GpuAcc};
    std::vector<armnn::BackendId> optOptions = {armnn::Compute::CpuRef};
    if (use_neon && use_opencl) {
      optOptions = {armnn::Compute::CpuAcc, armnn::Compute::GpuAcc};
    } else if (use_neon) {
      optOptions = {armnn::Compute::CpuAcc};
    } else if (use_opencl) {
      optOptions = {armnn::Compute::GpuAcc};
    }

    optNet = armnn::Optimize(*network, optOptions, runtime->GetDeviceSpec());

    // TODO: Get these from config
    std::string input_layer_name = "input";
    std::string output_layer_name = "output";
    inputBindingInfo = parser->GetNetworkInputBindingInfo(0, input_layer_name);
    outputBindingInfo =
        parser->GetNetworkOutputBindingInfo(0, output_layer_name);

    // get num of input and output nodes
    int numInputNodes = session->GetInputCount();
    assert(numInputNodes == inputBindingInfo.second.GetNumDimensions());
    int numOutputNodes = session->GetOutputCount();
    assert(numOutputNodes == outputBindingInfo.second.GetNumDimensions());

    armnn::TensorShape inShape = inputBindingInfo.second.GetShape();
    armnn::TensorShape outShape = outputBindingInfo.second.GetShape();

    // create dummy input buffers for device
    for (int i = 0; i < numInputNodes; ++i) {
      int64_t *raw_input = new int64_t[model_cfg->getInputSize(i)];
      buffers_in.push_back(raw_input);
    }

    // create dummy output buffers for device
    for (int i = 0; i < numOutputNodes; ++i) {
      float *raw_output = new float[model_cfg->getOutputSize(i)];
      buffers_out.push_back(raw_output);
    }
  }

  virtual int Inference(std::vector<Sample> samples) {

    // populate device input buffers from datasource
    model->configureWorkload(data_source, &samples, buffers_in);

    // do device specific inference here
    armnn::InputTensors inputTensor =
        MakeInputTensors(inputBindingInfo, buffers_in.data());
    armnn::OutputTensors outputTensor =
        MakeOutputTensors(outputBindingInfo, buffers_out.data());

    runtime->LoadNetwork(networkIdentifier, std::move(optNet));
    armnn::Status ret =
        runtime->EnqueueWorkload(networkIdentifier, inputTensor, outputTensor);

    // pass device output buffers to model specific post processing
    model->postprocessResults(&samples, buffers_out);

    return 0;
  }

  ~Device() { delete session; }

private:
  IModelConfig *model_cfg;

  // activations, set, input buffers
  std::vector<void *> buffers_in;

  // activation, set, output buffers
  std::vector<void *> buffers_out;

  IModel *model;
  IDataSource *data_source;

  size_t numInputNodes;
  size_t numOutputNodes;

  armnnTfLiteParser::BindingPointInfo inputBindingInfo;
  armnnTfLiteParser::BindingPointInfo outputBindingInfo;

  armnn::IRuntimePtr runtime;
  armnn::NetworkId networkIdentifier;
  armnn::IOptimizedNetworkPtr optNet;

  armnn::InputTensors MakeInputTensors(
      const std::pair<armnn::LayerBindingId, armnn::TensorInfo> &input,
      const void *inputTensorData) {
    return {{input.first, armnn::ConstTensor(input.second, inputTensorData)}};
  }

  armnn::OutputTensors MakeOutputTensors(
      const std::pair<armnn::LayerBindingId, armnn::TensorInfo> &output,
      void *outputTensorData) {
    return {{output.first, armnn::Tensor(output.second, outputTensorData)}};
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
