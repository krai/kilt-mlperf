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

#ifndef SQUAD_DATASOURCE_IMPL_H
#define SQUAD_DATASOURCE_IMPL_H

#include <stdio.h>
#include <stdlib.h>

#include "idatasource.h"

namespace KRAI {

template <typename TInputDataType> class BertDataSource : public IDataSource {
public:
  BertDataSource(const IConfig *config, std::vector<int> &affinities)
      : IDataSource(affinities), _config(config) {

    datasource_seq_len =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg)
            ->getDataSourceSequenceLength();

    // size for input ids, input mask, and segment ids.
    input_tensors.resize(3);
  }

  virtual ~BertDataSource() {}

  void loadSamplesImpl(void *user) override {

    SquadDataSourceConfig *datasource_config =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg);

    // load the input_ids
    loadTensors(datasource_config->getInputIDs(), input_tensors[0]);

    // load the input_masks
    loadTensors(datasource_config->getInputMask(), input_tensors[1]);

    // load the segment_ids
    loadTensors(datasource_config->getSegmentIDs(), input_tensors[2]);
  }

  void unloadSamples(void *user) override {}

  virtual void *getSamplePtr(int sample_idx, int buffer_idx) {

    int offset = sample_idx * datasource_seq_len;

    return &input_tensors[buffer_idx][offset];
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
  void loadTensors(std::string src_path, std::vector<TInputDataType> &t) {
    std::ifstream file(src_path, std::ios::in | std::ios::binary);
    if (!file)
      throw "Failed to open the file at " + src_path;
    file.seekg(0, std::ios::end);
    int size = file.tellg();
    file.seekg(0, std::ios::beg);

    using rawDataType = uint64_t;
    int vector_size = size / (sizeof(rawDataType));

    std::vector<rawDataType> _raw_data;
    _raw_data.resize(vector_size);
    file.read(reinterpret_cast<char *>(&_raw_data[0]), size);
    file.close();

    t.resize(vector_size);
    for (int i = 0; i < vector_size; ++i) {
      t[i] = static_cast<TInputDataType>(_raw_data[i]);
    }
  }

  const IConfig *_config;

  unsigned int datasource_seq_len;

  std::vector<std::vector<TInputDataType>> input_tensors;
};

} // namespace KRAI

#endif // SQUAD_DATASOURCE_IMPL_H
