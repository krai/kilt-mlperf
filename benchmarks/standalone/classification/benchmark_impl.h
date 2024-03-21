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

// Important: this has to be done before model_impl.h can be included
typedef mlperf::QuerySample Sample; // Type is superseded by sample.h in network

#include "kilt_impl.h"
#include "model_impl.h"
#include "datasource_impl.h"
#include "loadgen.h"

#if defined(__amd64__) && defined(ENABLE_ZEN2)
#include <cstdint>
#include <immintrin.h>
#endif

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {
IModel *modelConstruct(IConfig *config) {

  if (config->model_cfg->getInputDatatype(0) ==
          IModelConfig::IO_TYPE::FLOAT32 &&
      config->model_cfg->getOutputDatatype(0) == IModelConfig::IO_TYPE::FLOAT32)
    return new ResNet50Model<float, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::FLOAT32 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new ResNet50Model<float, int64_t>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::UINT8 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new ResNet50Model<uint8_t, int64_t>(config);
  else
    throw std::invalid_argument(
        "Input/output types not supported when constructing model");
}

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  if (config->model_cfg->getInputDatatype(0) ==
          IModelConfig::IO_TYPE::FLOAT32 &&
      config->model_cfg->getOutputDatatype(0) == IModelConfig::IO_TYPE::FLOAT32)
    return new ResNet50DataSource<float, float>(config, affinities);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::FLOAT32 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new ResNet50DataSource<float, int64_t>(config, affinities);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::UINT8 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new ResNet50DataSource<uint8_t, int64_t>(config, affinities);
  else
    throw std::invalid_argument(
        "Input/output types not supported when constructing datasource");
}

typedef KraiInferenceLibrary<mlperf::QuerySample> KILT;

} // namespace KRAI

#endif // BENCHMARK_IMPL_H