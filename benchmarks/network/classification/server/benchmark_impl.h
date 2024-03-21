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

#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include "kilt_impl.h"
#include "sample.h"

#include "benchmarks/standalone/classification/model_impl.h"

#include "config/benchmark_config.h"

#if defined(__amd64__) && defined(ENABLE_ZEN2)
#include <cstdint>
#include <immintrin.h>
#endif

namespace KRAI {

template <typename TInputDataType, typename TOutputDataType>
class Resnet50NetworkedModel
    : public ResNet50Model<TInputDataType, TOutputDataType> {
public:
  Resnet50NetworkedModel(const IConfig *config)
      : ResNet50Model<TInputDataType, TOutputDataType>(config),
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

  if (config->model_cfg->getInputDatatype(0) ==
          IModelConfig::IO_TYPE::FLOAT32 &&
      config->model_cfg->getOutputDatatype(0) == IModelConfig::IO_TYPE::FLOAT32)
    return new Resnet50NetworkedModel<float, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::FLOAT32 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new Resnet50NetworkedModel<float, int64_t>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::UINT8 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new Resnet50NetworkedModel<uint8_t, int64_t>(config);
  else
    throw std::invalid_argument(
        "Input/output types not supported when constructing model");
}

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {
  return nullptr;
}

} // namespace KRAI

#endif // BENCHMARK_IMPL_H