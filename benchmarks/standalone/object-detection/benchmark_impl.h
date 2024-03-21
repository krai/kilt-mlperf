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

typedef mlperf::QuerySample Sample;

#include "kilt_impl.h"
#include "datasource_impl.h"
#include "model_impl.h"

#include "config/benchmark_config.h"

namespace KRAI {

IModel *modelConstruct(IConfig *config) {

  if (config->model_cfg->getInputDatatype(0) == IModelConfig::IO_TYPE::FLOAT32)
    return new ObjectDetectionModel<float, float, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::INT8)
    return new ObjectDetectionModel<int8_t, float, float>(config);
  else
#if defined(MODEL_R34)
    return new ObjectDetectionModel<uint8_t, uint8_t, uint16_t>(config);
#elif defined(MODEL_RX50)
    return new ObjectDetectionModel<uint8_t, uint16_t, uint16_t>(config);
#else // MODEL_MV1
    return new ObjectDetectionModel<uint8_t, uint8_t, uint8_t>(config);
#endif
}

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  if (config->model_cfg->getInputDatatype(0) == IModelConfig::IO_TYPE::FLOAT32)
    return new ObjectDetectionDataSource<float>(config, affinities);
  else
    return new ObjectDetectionDataSource<uint8_t>(config, affinities);
}

typedef KraiInferenceLibrary<mlperf::QuerySample> KILT;

} // namespace KRAI

#endif // BENCHMARK_IMPL_H
