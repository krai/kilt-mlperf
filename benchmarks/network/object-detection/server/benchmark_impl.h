//
// MIT License
//
// Copyright (c) 2021 - 2023 Krai Ltd
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

#ifndef BENCHMARK_IMPL_H
#define BENCHMARK_IMPL_H

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "kilt_impl.h"
#include "sample.h"

#include "config/benchmark_config.h"

#if defined(MODEL_R34)
#define Model_Params R34_Params
#elif defined(MODEL_RX50)
#define Model_Params RX50_Params
#else
#define Model_Params MV1_Params
#endif

#include "benchmarks/standalone/object-detection/model_impl.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

template <typename TInputDataType, typename TOutput1DataType,
          typename TOutput2DataType>
class ObjectDetectionNetworkModel
    : public ObjectDetectionModel<TInputDataType, TOutput1DataType,
                                  TOutput2DataType> {

public:
  ObjectDetectionNetworkModel(const IConfig *config)
      : ObjectDetectionModel<TInputDataType, TOutput1DataType,
                             TOutput2DataType>(config),
        _config(config) {}

  virtual void *getSamplePtr(IDataSource *data_source, const Sample *s,
                             int buffer_idx) {
    return s->buf;
  }

  virtual void pushResult(Sample *sample, size_t size, float *result) {
    sample->Callback(size, result);
  }

private:
  const IConfig *_config;
};

IModel *modelConstruct(IConfig *config) {

  if (config->model_cfg->getInputDatatype(0) == IModelConfig::IO_TYPE::FLOAT32)
    return new ObjectDetectionNetworkModel<float, float, float>(config);
  else
#if defined(MODEL_R34)
    return new ObjectDetectionNetworkModel<uint8_t, uint8_t, uint16_t>(config);
#elif defined(MODEL_RX50)
    return new ObjectDetectionNetworkModel<uint8_t, uint16_t, uint16_t>(config);
#else // MODEL_MV1
    return new ObjectDetectionNetworkModel<uint8_t, uint8_t, uint8_t>(config);
#endif
}

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  return nullptr;
}

} // namespace KRAI

#endif // BENCHMARK_IMPL_H
