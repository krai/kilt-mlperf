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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.POSSIBILITY OF SUCH DAMAGE.
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
