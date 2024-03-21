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

#include <stdio.h>
#include <stdlib.h>

#include "config/benchmark_config.h"

typedef std::pair<mlperf::QuerySample, int> SizedSample;

#include "kilt_impl.h"
#include "datasource_impl.h"
#include "model_impl.h"

namespace KRAI {

IModel *modelConstruct(IConfig *config) {

  if (config->model_cfg->getInputDatatype(0) == IModelConfig::IO_TYPE::UINT64)
    return new BertModel<uint64_t, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::INT64)
    return new BertModel<int64_t, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::UINT32)
    return new BertModel<uint32_t, uint8_t>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::INT32)
    return new BertModel<int32_t, float>(config);
  else
    throw "Invalid data type for model construct";
}

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  if (config->model_cfg->getInputDatatype(0) == IModelConfig::IO_TYPE::UINT64)
    return new BertDataSource<uint64_t>(config, affinities);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::INT64)
    return new BertDataSource<int64_t>(config, affinities);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::UINT32)
    return new BertDataSource<uint32_t>(config, affinities);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::INT32)
    return new BertDataSource<int32_t>(config, affinities);
  else
    throw "Invalid data type for datasource construct";
}

class KILT : public KraiInferenceLibrary<SizedSample> {
public:
  void Inference(const std::vector<mlperf::QuerySample> &samples) {

    std::vector<SizedSample> sized_samples(samples.size());

    for (int i = 0; i < samples.size(); ++i) {
      sized_samples[i] = std::make_pair(samples[i], 0);
    }

    KraiInferenceLibrary<SizedSample>::Inference(sized_samples);
  }
};

} // namespace KRAI

#endif // BENCHMARK_IMPL_H
