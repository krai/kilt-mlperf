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

#ifndef DATASOURCE_H
#define DATASOURCE_H

#include <stdio.h>
#include <stdlib.h>

#include <atomic>

#include "loadgen.h"
#include "query_sample_library.h"

#include "config/benchmark_config.h"
#include "idatasource.h"

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

template <typename TInputDataType>
class ObjectDetectionDataSource : public IDataSource {
public:
  ObjectDetectionDataSource(const IConfig *config, std::vector<int> &affinities)
      : IDataSource(affinities), _config(config) {
    datasource_cfg =
        static_cast<ObjectDetectionDataSourceConfig *>(_config->datasource_cfg);
  }

  virtual ~ObjectDetectionDataSource() {}

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

    int vl = 0;

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
  ObjectDetectionDataSourceConfig *datasource_cfg;
  int _current_buffer_size = 0;
};

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities) {

  return new ObjectDetectionDataSource<uint64_t>(config, affinities);
}

} // namespace KRAI

#endif // DATASOURCE_H
