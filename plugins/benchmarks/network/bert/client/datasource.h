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

#ifndef BERT_H
#define BERT_H

#include <stdio.h>
#include <stdlib.h>

#include <atomic>

#include "loadgen.h"

#include "query_sample_library.h"

typedef std::pair<mlperf::QuerySample, int> SizedSample;

#include "idatasource.h"
#include "benchmark_config.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

template <typename TInputDataType> class BertDataSource : public IDataSource {
public:
  BertDataSource(const IConfig *config, std::vector<int> &affinities)
      : IDataSource(affinities), _config(config) {}

  virtual ~BertDataSource() {}

  void loadSamplesImpl(void *user) override {

    SquadDataSourceConfig *datasource_config =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg);
    // load the input_ids
    {
      std::ifstream file(datasource_config->getInputIDs(),
                         std::ios::in | std::ios::binary);
      if (!file)
        throw "Failed to open input_ids file " +
            datasource_config->getInputIDs();

      file.seekg(0, std::ios::end);
      int size = file.tellg();
      file.seekg(0, std::ios::beg);
      _input_ids.resize(size / (sizeof(uint64_t)));
      file.read(reinterpret_cast<char *>(&_input_ids[0]), size);
      file.close();
    }

    // load the input_masks
    {
      std::ifstream file(datasource_config->getInputMask(),
                         std::ios::in | std::ios::binary);
      if (!file)
        throw "Failed to open input_mask file " +
            datasource_config->getInputMask();

      file.seekg(0, std::ios::end);
      int size = file.tellg();
      file.seekg(0, std::ios::beg);
      _input_mask.resize(size / (sizeof(uint64_t)));
      file.read(reinterpret_cast<char *>(&_input_mask[0]), size);
      file.close();
    }

    // load the segment_ids
    {
      std::ifstream file(datasource_config->getSegmentIDs(),
                         std::ios::in | std::ios::binary);
      if (!file)
        throw "Failed to open segment_ids file " +
            datasource_config->getSegmentIDs();

      file.seekg(0, std::ios::end);
      int size = file.tellg();
      file.seekg(0, std::ios::beg);
      _segment_ids.resize(size / (sizeof(uint64_t)));
      file.read(reinterpret_cast<char *>(&_segment_ids[0]), size);
      file.close();
    }
  }

  void unloadSamples(void *user) override {}

  virtual void *getSamplePtr(int sample_idx, int buffer_idx) {

    int offset = sample_idx *
                 static_cast<SquadDataSourceConfig *>(_config->datasource_cfg)
                     ->getDataSourceSequenceLength();

    if (buffer_idx == 0)
      return &_input_ids[offset];
    if (buffer_idx == 1)
      return &_input_mask[offset];
    if (buffer_idx == 2)
      return &_segment_ids[offset];
    else
      throw "Invalid input pointer index.";
  }

  virtual const int getNumAvailableSampleFiles() {
    return static_cast<SquadDataSourceConfig *>(_config->datasource_cfg)
        ->getDatasetSize();
  };

  virtual const int getNumMaxSamplesInMemory() {
    return static_cast<SquadDataSourceConfig *>(_config->datasource_cfg)
        ->getBufferSize();
  };

private:
  const IConfig *_config;

  std::vector<int64_t> _input_ids;
  std::vector<int64_t> _input_mask;
  std::vector<int64_t> _segment_ids;
};

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  return new BertDataSource<uint64_t>(config, affinities);
}

} // namespace KRAI

#endif // BERT_H
