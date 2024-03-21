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

#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include "kilt_impl.h"
#include "loadgen.h"

#include "config/benchmark_config.h"

#if defined(__amd64__) && defined(ENABLE_ZEN2)
#include <cstdint>
#include <immintrin.h>
#endif

namespace KRAI {

template <typename TInputDataType, typename TOutputDataType>
class ResNet50Model : public IModel {
public:
  ResNet50Model(const IConfig *config) : _config(config) {
    datasource_cfg =
        static_cast<ClassificationDataSourceConfig *>(_config->datasource_cfg);

    input_buffer_size =
        datasource_cfg->getImageSize() * datasource_cfg->getImageSize() *
        datasource_cfg->getNumChannels() * sizeof(TInputDataType);
  }

  void configureWorkload(IDataSource *data_source, void *device,
                         const void *samples, std::vector<void *> &in_ptrs) {

    const std::vector<Sample> *s =
        reinterpret_cast<const std::vector<Sample> *>(samples);

    IDevice<Sample> *d = reinterpret_cast<IDevice<Sample> *>(device);

    TInputDataType *dest_ptr = reinterpret_cast<TInputDataType *>(in_ptrs[0]);

    for (int i = 0; i < s->size(); ++i) {

      TInputDataType *src_ptr = reinterpret_cast<TInputDataType *>(
          getSamplePtr(data_source, &(*s)[i], 0));

      d->SyncData(src_ptr, dest_ptr, i * input_buffer_size, input_buffer_size);
    }
  }

  void postprocessResults(void *samples, std::vector<void *> &out_ptrs) {

    int probe_offset = datasource_cfg->getHasBackgroundClass() ? 1 : 0;

    std::vector<Sample> *s = reinterpret_cast<std::vector<Sample> *>(samples);

    float encoding_buffer[s->size()];

    for (int i = 0; i < s->size(); ++i) {

      TOutputDataType *ptr = (TOutputDataType *)out_ptrs[0] + i;

      encoding_buffer[i] = (float)*ptr - probe_offset;
      pushResult(&(*s)[i], 1, &encoding_buffer[i]);
    }
  };

private:
  virtual void *getSamplePtr(IDataSource *data_source, const Sample *s,
                             int buffer_idx) {
    return data_source->getSamplePtr(s->index, buffer_idx);
  }

  virtual void pushResult(Sample *sample, size_t size, float *result) {
#ifdef STANDALONE
    mlperf::QuerySampleResponse response(
        {sample->id, uintptr_t(result), sizeof(float) * size});
    mlperf::QuerySamplesComplete(&response, 1);
#endif
  }

  const IConfig *_config;
  ClassificationDataSourceConfig *datasource_cfg;
  int input_buffer_size;
};

} // namespace KRAI
