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
#include "config/kilt_config.h"

#include "kilt_impl.h"

#include "pack.h"

#include "benchmarks/standalone/bert/model_impl.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

template <typename TInputDataType, typename TOutputDataType>
class BertNetworkModel : public BertModel<TInputDataType, TOutputDataType> {
public:
  BertNetworkModel(const IConfig *config)
      : BertModel<TInputDataType, TOutputDataType>(config), _config(config) {}

  virtual void *getSamplePtr(IDataSource *data_source, const SizedSample *s,
                             int sample_idx, int buffer_idx) {
    if (buffer_idx == 0)
      return s->first.buf0;
    else if (buffer_idx == 1)
      return s->first.buf1;
    else // must be buffer_idx = 2
      return s->first.buf2;
  }

  virtual void pushResult(SizedSample *sample, std::vector<float> &result) {
    sample->first.Callback(&result[0]);
  }

private:
  const IConfig *_config;
};

IModel *modelConstruct(IConfig *config) {

  if (config->model_cfg->getInputDatatype(0) == IModelConfig::IO_TYPE::UINT64)
    return new BertNetworkModel<uint64_t, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::INT64)
    return new BertNetworkModel<int64_t, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::UINT32)
    return new BertNetworkModel<uint32_t, uint8_t>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
           IModelConfig::IO_TYPE::INT32)
    return new BertNetworkModel<int32_t, float>(config);
  else
    throw "Invalid data type for model construct";
}

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  return nullptr;
}

} // namespace KRAI

#endif // BENCHMARK_IMPL_H
