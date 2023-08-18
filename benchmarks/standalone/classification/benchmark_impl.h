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

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

template <typename TData> class StaticBuffer {
public:
  StaticBuffer(const int size, TData *buffer = nullptr) : _size(size) {
    if (buffer != nullptr)
      _buffer = buffer;
    else
      _buffer = (TData *)aligned_alloc(32, size * sizeof(TData));
  }

  virtual ~StaticBuffer() { free(_buffer); }

  TData *data() const { return _buffer; }
  int size() const { return _size; }

protected:
  const int _size;
  TData *_buffer;
};

template <typename TData> class SampleData : public StaticBuffer<TData> {
public:
  SampleData(const int s, TData *buffer = nullptr)
      : StaticBuffer<TData>(s, buffer) {}

  virtual void load(const std::string &path, int vl) {

    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file)
      throw "Failed to open image data " + path;
    file.read(reinterpret_cast<char *>(this->_buffer),
              this->_size * sizeof(TData));
    if (!file)
      std::cerr << "error: only " << file.gcount() << " could be read"
                << std::endl;
    if (vl > 1) {
      std::cout << "Loaded file: " << path << std::endl;
    } else if (vl) {
      std::cout << 'l' << std::flush;
    }
  }
};

template <typename TInputDataType, typename TOutputDataType>
class ResNet50Model : public IModel {
public:
  ResNet50Model(const IConfig *config) : _config(config) {
    datasource_cfg =
        static_cast<ClassificationDataSourceConfig *>(_config->datasource_cfg);
  }

  void configureWorkload(IDataSource *data_source, const void *samples,
                         std::vector<void *> &in_ptrs) override {

    const std::vector<mlperf::QuerySample> *s =
        reinterpret_cast<const std::vector<mlperf::QuerySample> *>(samples);

    uint32_t buf_size = datasource_cfg->getImageSize() *
                        datasource_cfg->getImageSize() *
                        datasource_cfg->getNumChannels();

    for (int i = 0; i < s->size(); ++i) {

      TInputDataType *src_ptr = reinterpret_cast<TInputDataType *>(
          data_source->getSamplePtr((*s)[i].index, 0));

      TInputDataType *dest_ptr =
          reinterpret_cast<TInputDataType *>(in_ptrs[0]) + i * buf_size;

#if defined(__amd64__) && defined(ENABLE_ZEN2)
      const __m256i *src = reinterpret_cast<const __m256i *>(src_ptr);
      __m256i *dest = reinterpret_cast<__m256i *>(dest_ptr);
      int64_t vectors = buf_size / sizeof(*src);
      for (; vectors > 0; vectors--, src++, dest++) {
        const __m256i loaded = _mm256_stream_load_si256(src);
        _mm256_stream_si256(dest, loaded);
      }
      unsigned rem = buf_size % sizeof(*src);
      if (rem > 0) {
        memcpy((uint8_t *)dest, (uint8_t *)src, rem);
      }
      _mm_sfence();
#else
      std::copy(src_ptr, src_ptr + buf_size, dest_ptr);
#endif
    }
  }

  void postprocessResults(void *samples, std::vector<void *> &out_ptrs) {

    int probe_offset = datasource_cfg->getHasBackgroundClass() ? 1 : 0;

    std::vector<mlperf::QuerySample> *s =
        reinterpret_cast<std::vector<mlperf::QuerySample> *>(samples);

    std::vector<mlperf::QuerySampleResponse> responses;

    responses.reserve(s->size());

    float encoding_buffer[s->size()];

    for (int i = 0; i < s->size(); ++i) {

      TOutputDataType *ptr = (TOutputDataType *)out_ptrs[0] + i;

      encoding_buffer[i] = (float)*ptr - probe_offset;
      responses.push_back(
          {(*s)[i].id, uintptr_t(&encoding_buffer[i]), sizeof(float)});
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  };

private:
  const IConfig *_config;
  ClassificationDataSourceConfig *datasource_cfg;
  int _current_buffer_size = 0;
};

IModel *modelConstruct(IConfig *config) {

  if (config->model_cfg->getInputDatatype(0) ==
          IModelConfig::IO_TYPE::FLOAT32 &&
      config->model_cfg->getOutputDatatype(0) == IModelConfig::IO_TYPE::FLOAT32)
    return new ResNet50Model<float, float>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::FLOAT32 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new ResNet50Model<float, int64_t>(config);
  else if (config->model_cfg->getInputDatatype(0) ==
               IModelConfig::IO_TYPE::UINT8 &&
           config->model_cfg->getOutputDatatype(0) ==
               IModelConfig::IO_TYPE::INT64)
    return new ResNet50Model<uint8_t, int64_t>(config);
  else
    throw std::invalid_argument(
        "Input/output types not supported when constructing model");
}

template <typename TInputDataType, typename TOutputDataType>
class ResNet50DataSource : public IDataSource {
public:
  ResNet50DataSource(const IConfig *config, std::vector<int> &affinities)
      : IDataSource(affinities), _config(config) {
    datasource_cfg =
        static_cast<ClassificationDataSourceConfig *>(_config->datasource_cfg);
  }

  virtual ~ResNet50DataSource() {}

  const std::vector<std::string> &
  loadFilenames(std::vector<size_t> img_indices) {
    _filenames_buffer.clear();
    _filenames_buffer.reserve(img_indices.size());
    idx2loc.clear();

    auto list_of_available_imagefiles =
        datasource_cfg->getListOfImageFilenames();
    auto count_available_imagefiles = list_of_available_imagefiles.size();

    int loc = 0;
    for (auto idx : img_indices) {
      if (idx < count_available_imagefiles) {
        _filenames_buffer.emplace_back(list_of_available_imagefiles[idx]);
        idx2loc[idx] = loc++;
      } else {
        std::cerr << "Trying to load filename[" << idx << "] when only "
                  << count_available_imagefiles << " images are available"
                  << std::endl;
        exit(1);
      }
    }

    return _filenames_buffer;
  }

  void loadSamplesImpl(void *user) override {

    const std::vector<size_t> *img_indices =
        static_cast<const std::vector<size_t> *>(user);

    loadFilenames(*img_indices);

    auto vl = _config->server_cfg->getVerbosity();

    unsigned length = _filenames_buffer.size();
    _current_buffer_size = length;
    _in_batch = new std::unique_ptr<SampleData<TInputDataType>>[length];
    unsigned batch_size = _config->model_cfg->getBatchSize();
    unsigned image_size = datasource_cfg->getImageSize() *
                          datasource_cfg->getImageSize() *
                          datasource_cfg->getNumChannels();

    for (auto i = 0; i < length; i += batch_size) {
      unsigned actual_batch_size =
          std::min(batch_size, batch_size < length ? (length - i) : length);
      TInputDataType *buf = (TInputDataType *)aligned_alloc(
          32, batch_size * image_size * sizeof(TInputDataType));
      for (auto j = 0; j < actual_batch_size; j++, buf += image_size) {
        _in_batch[i + j].reset(new SampleData<TInputDataType>(image_size, buf));
        std::string path =
            datasource_cfg->getDatasetDir() + "/" + _filenames_buffer[i + j];
        _in_batch[i + j]->load(path, vl);
      }
    }
  }

  void unloadSamples(void *user) override {
    unsigned num_examples = _filenames_buffer.size();
    uint16_t batch_size = _config->model_cfg->getBatchSize();
    for (size_t i = 0; i < num_examples; i += batch_size) {
      delete _in_batch[i].get();
    }
  }

  virtual void *getSamplePtr(int img_idx, int) {
    return _in_batch[idx2loc[img_idx]].get()->data();
  }

  virtual const int getNumAvailableSampleFiles() {
    return datasource_cfg->getListOfImageFilenames().size();
  };

  virtual const int getNumMaxSamplesInMemory() {
    return datasource_cfg->getMaxImagesInMemory();
  };

  std::map<int, int> idx2loc;

private:
  const IConfig *_config;
  std::vector<std::string> _filenames_buffer;
  std::unique_ptr<SampleData<TInputDataType>> *_in_batch;
  ClassificationDataSourceConfig *datasource_cfg;
  int _current_buffer_size = 0;
};

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

typedef KraiInferenceLibrary<mlperf::QuerySample> KILT;

} // namespace KRAI

#endif // BENCHMARK_IMPL_H
