//
// MIT License
//
// Copyright (c) 2023 - 2023 Krai Ltd
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

#include <cuda_fp16.h>

#include "config/benchmark_config.h"

#include "loadgen.h"
#include "query_sample_library.h"

typedef std::pair<mlperf::QuerySample, int> SizedSample;

#include "kilt_impl.h"

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

    int incoming_batch_size = sm.size();

    auto ds_config =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg);
    int dataset_seq_len = ds_config->getDataSourceSequenceLength();
    int device_batch_size = ds_config->getDeviceBatchSize();

    // get the sizes of each of the inputs
    for (int s = 0; s < incoming_batch_size; ++s) {
      // get the input mask
      TInputDataType *src = static_cast<TInputDataType *>(
          data_source->getSamplePtr(sm[s].first.index, 1));
      int sum = 0;
      for (int j = 0; j < dataset_seq_len; ++j) {
        sum += src[j];
      }
      sm[s].second = sum;
    }

    std::sort(sm.begin(), sm.end(),
              [](const SizedSample &a, const SizedSample &b) {
                return a.second > b.second; // sort in descending order
              });

    for (int s = 0; s < incoming_batch_size; s += device_batch_size) {
      // Determine the range of samples for the current batch
      int batch_end = std::min(s + device_batch_size, incoming_batch_size);
      std::vector<SizedSample> batch(sm.begin() + s, sm.begin() + batch_end);
      // Call the callback function for the current batch
      callback(handle, &batch);
    }
  }

  void configureWorkload(IDataSource *data_source, const void *samples,
                         std::vector<void *> &in_ptrs) {

    const std::vector<SizedSample> sm =
        *(reinterpret_cast<const std::vector<SizedSample> *>(samples));

    auto bm_config = static_cast<BertModelConfig *>(_config->model_cfg);
    unsigned int seq_len = bm_config->getModelSequenceLength();
    unsigned int num_samples = sm.size();

    // clear input buffers
    int inputCount = bm_config->getInputCount();
    for (int i = 0; i < inputCount; i++) {
      memset(in_ptrs[i], 0,
             bm_config->getInputSize(i) * sizeof(TInputDataType));
    }

    std::string engineSource = bm_config->getEngineSource();
    int offset = 0;

    for (int i = 0; i < num_samples; ++i) {
      TInputDataType *src0 = static_cast<TInputDataType *>(
          data_source->getSamplePtr(sm[i].first.index, 0));
      TInputDataType *src1 = static_cast<TInputDataType *>(
          data_source->getSamplePtr(sm[i].first.index, 1));
      TInputDataType *src2 = static_cast<TInputDataType *>(
          data_source->getSamplePtr(sm[i].first.index, 2));

      int sample_seq_len = sm[i].second;

      if (engineSource == "nvidia") {
        for (int m = 0; m < seq_len; m++) {
          static_cast<TInputDataType *>(in_ptrs[0])[m + offset] =
              src0[m]; // input_ids
          static_cast<TInputDataType *>(in_ptrs[1])[m + offset] =
              src2[m]; // segment_ids
        }
        static_cast<TInputDataType *>(in_ptrs[2])[i + 1] =
            static_cast<TInputDataType *>(in_ptrs[2])[i] + sample_seq_len;
      } else if (engineSource == "trtexec") {
        for (int m = 0; m < seq_len; m++) {
          static_cast<TInputDataType *>(in_ptrs[0])[m + offset] = src0[m];
          static_cast<TInputDataType *>(in_ptrs[1])[m + offset] = src1[m];
          static_cast<TInputDataType *>(in_ptrs[2])[m + offset] = src2[m];
        }
      }
      offset += sample_seq_len;
    }
  }

  void postprocessResults(void *samples, std::vector<void *> &out_ptrs) {

    std::vector<SizedSample> sm =
        *(reinterpret_cast<std::vector<SizedSample> *>(samples));
    unsigned int num_samples = sm.size();

    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(num_samples);

    std::vector<std::vector<float>> results;
    results.resize(num_samples);

    auto bm_config = static_cast<BertModelConfig *>(_config->model_cfg);
    std::string engineSource = bm_config->getEngineSource();

    if (engineSource == "nvidia") {
      TOutputDataType *outputData = static_cast<TOutputDataType *>(out_ptrs[0]);
      int offset = 0;

      for (int i = 0; i < num_samples; ++i) {
        int sample_seq_len = sm[i].second;
        results[i].resize(sample_seq_len * 2, -10000.0f);
        for (int j = 0; j < sample_seq_len * 2; ++j) {
          results[i][j] = static_cast<float>(*(outputData + offset + j));
        }
        offset += sample_seq_len * 2;

        responses.push_back({sm[i].first.id, uintptr_t(&results[i][0]),
                             sizeof(float) * results[i].size()});
      }
    } else if (engineSource == "trtexec") {
      int offset = 0;

      for (int i = 0; i < sm.size(); ++i) {

        int sample_seq_len = sm[i].second;

        results[i].resize(sample_seq_len * 2, -10000.0f);
        float *b0 = (float *)(out_ptrs[0]) + offset;
        float *b1 = (float *)(out_ptrs[1]) + offset;
        for (int j = 0; j < sample_seq_len; ++j) {
          results[i][j * 2] = *(b0 + j);
          results[i][(j * 2) + 1] = *(b1 + j);
        }
        offset += sample_seq_len;

        responses.push_back({sm[i].first.id, uintptr_t(&results[i][0]),
                             sizeof(float) * results[i].size()});
      }
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

  const IConfig *_config;
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
           IModelConfig::IO_TYPE::INT32) {
    if (config->model_cfg->getOutputDatatype(0) == IModelConfig::IO_TYPE::HALF)
      return new BertModel<int32_t, __half>(config);
    else if (config->model_cfg->getOutputDatatype(0) ==
             IModelConfig::IO_TYPE::FLOAT32)
      return new BertModel<int32_t, float>(config);
    else
      throw "Invalid data type for model construct";
  } else
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
