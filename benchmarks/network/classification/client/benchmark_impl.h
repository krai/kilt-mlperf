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

#pragma once

#include "config/benchmark_config.h"
#include "benchmarks/standalone/classification/datasource_impl.h"
#include "idevice.h"

namespace KRAI {
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
} // namespace KRAI

#include "benchmarks/network/common/client.h"
