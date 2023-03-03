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

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "loadgen.h"
#include "kilt.h"
#include "device.h"

#include "benchmark_config.h"
#include "kilt_config.h"
#include "device_config.h"

#if defined(__amd64__) && defined(ENABLE_ZEN2)
#include <immintrin.h>
#include <cstdint>
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
    file.read(reinterpret_cast<char *>(this->_buffer), this->_size);
    if (vl > 1) {
      std::cout << "Loaded file: " << path << std::endl;
    } else if (vl) {
      std::cout << 'l' << std::flush;
    }
  }
};

template <typename TInputDataType> class ResNet50Model : public IModel {
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

      int64_t *ptr = (int64_t *)out_ptrs[0] + i;

      encoding_buffer[i] = (float)*ptr - probe_offset;
      responses.push_back(
          {(*s)[i].id, uintptr_t(&encoding_buffer[i]), sizeof(float) });
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  };

private:
  const IConfig *_config;
  ClassificationDataSourceConfig *datasource_cfg;
  int _current_buffer_size = 0;
};

IModel *modelConstruct(IConfig *config) {

  if (config->device_cfg->getSkipStage() != "convert")
    return new ResNet50Model<float>(config);
  else
    return new ResNet50Model<uint8_t>(config);
}

template <typename TInputDataType>
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
    _in_batch = new std::unique_ptr<SampleData<TInputDataType> >[length];
    unsigned batch_size = _config->device_cfg->getBatchSize();
    unsigned image_size = datasource_cfg->getImageSize() *
                          datasource_cfg->getImageSize() *
                          datasource_cfg->getNumChannels();

    for (auto i = 0; i < length; i += batch_size) {
      unsigned actual_batch_size =
          std::min(batch_size, batch_size < length ? (length - i) : length);
      TInputDataType *buf =
          (TInputDataType *)aligned_alloc(32, batch_size * image_size);
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
    uint16_t batch_size = _config->device_cfg->getBatchSize();
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
  std::unique_ptr<SampleData<TInputDataType> > *_in_batch;
  ClassificationDataSourceConfig *datasource_cfg;
  int _current_buffer_size = 0;
};

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  if (config->device_cfg->getSkipStage() != "convert")
    return new ResNet50DataSource<float>(config, affinities);
  else
    return new ResNet50DataSource<uint8_t>(config, affinities);
}

typedef KraiInferenceLibrary<mlperf::QuerySample> KILT;

} // namespace KRAI

#endif // BENCHMARK_IMPL_H
