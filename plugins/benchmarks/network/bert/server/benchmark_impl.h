//
// Copyright (c) 2021-2023 Krai Ltd.
//
// SPDX-License-Identifier: BSD-3-Clause.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef BENCHMARK_IMPL_H
#define BENCHMARK_IMPL_H

#include <stdio.h>
#include <stdlib.h>

#include "benchmark_config.h"
#include "kilt_config.h"
#include "device_config.h"

#include "kilt.h"
#include "device.h"

#include "pack.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

template <typename TInputDataType, typename TOutputDataType>
class BertModel : public IModel {
public:
  BertModel(const IConfig *config) : _config(config) {}

  virtual void
  preprocessSamples(IDataSource *data_source, const void *samples, void *handle,
                    void (*callback)(void *handle, const void *samples)) {

    std::vector<SizedSample> sm =
        *(reinterpret_cast<const std::vector<SizedSample> *>(samples));

    unsigned int dataset_seq_len = static_cast<SquadDataSourceConfig *>(
        _config->datasource_cfg)->getDataSourceSequenceLength();
    unsigned int packed_seq_len = static_cast<BertModelConfig *>(
        _config->model_cfg)->getModelSequenceLength();

    // get the sizes of each of the inputs
    for (int s = 0; s < sm.size(); ++s) {

      // get the input mask
      uint64_t *src = sm[s].first.buf1;

      int sum = 0;
      for (int j = 0; j < dataset_seq_len; ++j) {
        sum += src[j];
      }
      sm[s].second = sum;
    }

    std::vector<std::vector<SizedSample> > packed_samples;

    pack(sm, packed_seq_len, 3, packed_samples);

    for (int ps = 0; ps < packed_samples.size(); ++ps) {
      callback(handle, &packed_samples[ps]);
    }
  }

  void configureWorkload(IDataSource *data_source, const void *samples,
                         std::vector<void *> &in_ptrs) override {

    const std::vector<SizedSample> *sm =
        reinterpret_cast<const std::vector<SizedSample> *>(samples);

    sample_count += sm->size();
    sample_delta += sm->size();

    unsigned int packed_seq_len = static_cast<BertModelConfig *>(
        _config->model_cfg)->getModelSequenceLength();

    // clear input buffers
    memset(in_ptrs[0], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[1], 0, 8 * sizeof(TInputDataType));
    memset(in_ptrs[2], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[3], 0, packed_seq_len * sizeof(TInputDataType));

    unsigned int offset = 0;

    for (int s = 0; s < sm->size(); ++s) {

      uint64_t *src0 = (*sm)[s].first.buf0;
      uint64_t *src2 = (*sm)[s].first.buf2;

      TInputDataType sample_seq_len = (*sm)[s].second;

      for (int m = 0; m < sample_seq_len; m++) {
        static_cast<TInputDataType *>(in_ptrs[0])[offset] = src0[m];
        static_cast<TInputDataType *>(in_ptrs[2])[offset] = src2[m];
        static_cast<TInputDataType *>(in_ptrs[3])[offset] = m;
        ++offset;
      }

      static_cast<TInputDataType *>(in_ptrs[1])[s] = sample_seq_len;
    }
  }

  void postprocessResults(void *samples, std::vector<void *> &out_ptrs) {

    std::vector<SizedSample> *sm =
        reinterpret_cast<std::vector<SizedSample> *>(samples);

    sample_count -= sm->size();

    unsigned int seq_len = static_cast<SquadDataSourceConfig *>(
        _config->datasource_cfg)->getDataSourceSequenceLength();

    // std::vector<mlperf::QuerySampleResponse> responses;
    // responses.reserve(sm->size());

    std::vector<std::vector<float> > results;
    results.resize(sm->size());

    int offset = 0;

    for (int i = 0; i < sm->size(); ++i) {

      TInputDataType sample_seq_len = (*sm)[i].second;

      results[i].resize(seq_len * 2, -10000.0f);
      TOutputDataType *b0 = ((TOutputDataType *)out_ptrs[0]) + offset;
      TOutputDataType *b1 = ((TOutputDataType *)out_ptrs[1]) + offset;
      for (int j = 0; j < sample_seq_len; ++j) {
        results[i][j * 2] = *(b0 + j);
        results[i][(j * 2) + 1] = *(b1 + j);
      }
      offset += sample_seq_len;

      (*sm)[i].first.callback(&((*sm)[i].first), &results[i][0]);
    }
  }

private:
  std::atomic<int> sample_count = 0;
  std::atomic<int> sample_delta = 0;
  const IConfig *_config;
};

IModel *modelConstruct(IConfig *config) {

  if (config->device_cfg->getSkipStage() != "convert")
    return new BertModel<uint64_t, float>(config);
  else
    return new BertModel<uint32_t, uint8_t>(config);
}

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  return nullptr;
}

} // namespace KRAI

#endif // BENCHMARK_IMPL_H
