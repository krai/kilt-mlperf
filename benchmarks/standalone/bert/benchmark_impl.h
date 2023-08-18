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

#include "loadgen.h"
#include "query_sample_library.h"

typedef std::pair<mlperf::QuerySample, int> SizedSample;

#include "kilt_impl.h"

#include "pack.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

template <typename TInputDataType, typename TOutputDataType>
class BertModel : public IModel {
public:
  BertModel(const IConfig *config) : _config(config) {

    distilbert = _config->model_cfg->getInputCount() == 3;
  }

  virtual void
  preprocessSamples(IDataSource *data_source, const void *samples, void *handle,
                    void (*callback)(void *handle, const void *samples)) {

    std::vector<SizedSample> sm =
        *(reinterpret_cast<const std::vector<SizedSample> *>(samples));

    unsigned int dataset_seq_len =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg)
            ->getDataSourceSequenceLength();
    unsigned int packed_seq_len =
        static_cast<BertModelConfig *>(_config->model_cfg)
            ->getModelSequenceLength();

    // get the sizes of each of the inputs
    for (int s = 0; s < sm.size(); ++s) {

      // get the input mask
      TInputDataType *src = static_cast<TInputDataType *>(
          data_source->getSamplePtr(sm[s].first.index, 1));

      int sum = 0;
      for (int j = 0; j < dataset_seq_len; ++j) {
        sum += src[j];
      }
      sm[s].second = sum;
    }

    std::vector<std::vector<SizedSample>> packed_samples;

    pack(sm, packed_seq_len, 3, packed_samples);

    for (int ps = 0; ps < packed_samples.size(); ++ps) {
      callback(handle, &packed_samples[ps]);
    }
  }

  void configureWorkload(IDataSource *data_source, const void *samples,
                         std::vector<void *> &in_ptrs) override {

    BertModelConfig::BERT_MODEL_VARIANT bmv =
        static_cast<BertModelConfig *>(_config->model_cfg)->getModelVariant();

    // TODO: convert this to a function pointer.
    if (bmv == BertModelConfig::BERT_ORIG)
      configureWorkloadOrig(data_source, samples, in_ptrs);
    else if (bmv == BertModelConfig::BERT_PACKED)
      configureWorkloadPacked(data_source, samples, in_ptrs);
    else // if (bmv == BertModelConfig::DISTILBERT_PACKED)
      configureWorkloadDistilBERTPacked(data_source, samples, in_ptrs);
  }

  void configureWorkloadOrig(IDataSource *data_source, const void *samples,
                             std::vector<void *> &in_ptrs) {

    const std::vector<SizedSample> *sm =
        reinterpret_cast<const std::vector<SizedSample> *>(samples);

    sample_count += sm->size();
    sample_delta += sm->size();

    unsigned int packed_seq_len =
        static_cast<BertModelConfig *>(_config->model_cfg)
            ->getModelSequenceLength();

    packed_seq_len = 384;

    // clear input buffers
    memset(in_ptrs[0], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[1], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[2], 0, packed_seq_len * sizeof(TInputDataType));

    TInputDataType *src0 = static_cast<TInputDataType *>(
        data_source->getSamplePtr((*sm)[0].first.index, 0));
    TInputDataType *src1 = static_cast<TInputDataType *>(
        data_source->getSamplePtr((*sm)[0].first.index, 1));
    TInputDataType *src2 = static_cast<TInputDataType *>(
        data_source->getSamplePtr((*sm)[0].first.index, 2));

    TInputDataType sample_seq_len = (*sm)[0].second;

    for (int m = 0; m < sample_seq_len; m++) {
      static_cast<TInputDataType *>(in_ptrs[0])[m] = src0[m];
      static_cast<TInputDataType *>(in_ptrs[1])[m] = src1[m];
      static_cast<TInputDataType *>(in_ptrs[2])[m] = src2[m];
    }
  }

  void configureWorkloadPacked(IDataSource *data_source, const void *samples,
                               std::vector<void *> &in_ptrs) {

    const std::vector<SizedSample> *sm =
        reinterpret_cast<const std::vector<SizedSample> *>(samples);

    sample_count += sm->size();
    sample_delta += sm->size();

    unsigned int packed_seq_len =
        static_cast<BertModelConfig *>(_config->model_cfg)
            ->getModelSequenceLength();

    // clear input buffers
    memset(in_ptrs[0], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[1], 0, 8 * sizeof(TInputDataType));
    memset(in_ptrs[2], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[3], 0, packed_seq_len * sizeof(TInputDataType));

    unsigned int offset = 0;

    for (int s = 0; s < sm->size(); ++s) {

      TInputDataType *src0 = static_cast<TInputDataType *>(
          data_source->getSamplePtr((*sm)[s].first.index, 0));
      TInputDataType *src2 = static_cast<TInputDataType *>(
          data_source->getSamplePtr((*sm)[s].first.index, 2));

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

  void configureWorkloadDistilBERTPacked(IDataSource *data_source,
                                         const void *samples,
                                         std::vector<void *> &in_ptrs) {

    const std::vector<SizedSample> *sm =
        reinterpret_cast<const std::vector<SizedSample> *>(samples);

    sample_count += sm->size();
    sample_delta += sm->size();

    unsigned int packed_seq_len =
        static_cast<BertModelConfig *>(_config->model_cfg)
            ->getModelSequenceLength();

    // clear input buffers
    memset(in_ptrs[0], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[1], 0,
           packed_seq_len * packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[2], 0, packed_seq_len * sizeof(TInputDataType));

    unsigned int offset = 0;

    for (int s = 0; s < sm->size(); ++s) {

      TInputDataType *src0 = static_cast<TInputDataType *>(
          data_source->getSamplePtr((*sm)[s].first.index, 0));

      TInputDataType sample_seq_len = (*sm)[s].second;

      apply_mask(static_cast<TInputDataType *>(in_ptrs[1]), sample_seq_len,
                 offset);

      for (int m = 0; m < sample_seq_len; m++) {
        static_cast<TInputDataType *>(in_ptrs[0])[offset] = src0[m];
        static_cast<TInputDataType *>(in_ptrs[2])[offset] = m;
        ++offset;
      }
    }
  }

  void postprocessResults(void *samples, std::vector<void *> &out_ptrs) {

    std::vector<SizedSample> *sm =
        reinterpret_cast<std::vector<SizedSample> *>(samples);

    sample_count -= sm->size();

    unsigned int seq_len =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg)
            ->getDataSourceSequenceLength();

    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(sm->size());

    std::vector<std::vector<float>> results;
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

      responses.push_back({(*sm)[i].first.id, uintptr_t(&results[i][0]),
                           sizeof(float) * results[i].size()});
    }

    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  };

private:
  void apply_mask(TInputDataType *ptr, int seq_len, int offset) {

    unsigned int packed_seq_len =
        static_cast<BertModelConfig *>(_config->model_cfg)
            ->getModelSequenceLength();

    // add the y offset
    ptr += offset * packed_seq_len;

    // create the mask for the first line
    for (int i = offset; i < seq_len + offset; ++i)
      ptr[i] = 1;

    // replicate in the y direction
    for (int i = 1; i < seq_len; ++i) {
      TInputDataType *src = ptr + offset;
      TInputDataType *dst = src + packed_seq_len * i;
      memcpy(dst, src, seq_len * sizeof(TInputDataType));
    }
  }

  std::atomic<int> sample_count = 0;
  std::atomic<int> sample_delta = 0;
  const IConfig *_config;

  bool distilbert;
};

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

template <typename TInputDataType> class BertDataSource : public IDataSource {
public:
  BertDataSource(const IConfig *config, std::vector<int> &affinities)
      : IDataSource(affinities), _config(config) {}

  virtual ~BertDataSource() {}

  void loadSamplesImpl(void *user) override {

    SquadDataSourceConfig *datasource_config =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg);

    // load the input_ids
    loadVector(datasource_config->getInputIDs(), _input_ids);

    // load the input_masks
    loadVector(datasource_config->getInputMask(), _input_mask);

    // load the segment_ids
    loadVector(datasource_config->getSegmentIDs(), _segment_ids);
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
  void loadVector(std::string src_path, std::vector<TInputDataType> &vector) {
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

    vector.resize(vector_size);
    for (int i = 0; i < vector_size; ++i) {
      vector[i] = static_cast<TInputDataType>(_raw_data[i]);
    }
  }

  const IConfig *_config;

  std::vector<TInputDataType> _input_ids;
  std::vector<TInputDataType> _input_mask;
  std::vector<TInputDataType> _segment_ids;
};

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
