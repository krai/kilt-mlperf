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

#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#pragma once

#include <fstream>
#include <sstream>

#include "config/config_tools/config_tools.h"
#include "iconfig.h"

namespace KRAI {

//----------------------------------------------------------------------

class ClassificationDataSourceConfig : public IDataSourceConfig {

public:
  virtual const int getImageSize() const { return image_size; }

  virtual const int getNumChannels() const { return num_channels; }

  virtual const bool getHasBackgroundClass() const {
    return has_background_class;
  }

  virtual const std::vector<std::string> &getListOfImageFilenames() const {
    return _available_image_list;
  }

  virtual const std::string &getDatasetDir() const { return images_dir; }

  virtual const int getMaxImagesInMemory() const {
    return images_in_memory_max;
  }

  ClassificationDataSourceConfig() {

    std::ifstream file(available_images_file);
    if (!file)
      throw "Unable to open the available image list file " +
          available_images_file;
    for (std::string s; !getline(file, s).fail();)
      _available_image_list.emplace_back(s);
    std::cout << "Number of available imagefiles: "
              << _available_image_list.size() << std::endl;
  }

private:
  const int image_size =
      getconfig_i("KILT_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE");

  const int num_channels = 3;

  const bool has_background_class =
      getconfig_s("KILT_DATASET_IMAGENET_HAS_BACKGROUND_CLASS") == "YES";

  const std::string available_images_file =
      getconfig_s("KILT_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF");

  const std::string images_dir =
      getconfig_s("KILT_DATASET_IMAGENET_PREPROCESSED_DIR");

  const int images_in_memory_max = getconfig_i("LOADGEN_BUFFER_SIZE");

  std::vector<std::string> _available_image_list;
};

IDataSourceConfig *getDataSourceConfig() {
  return new ClassificationDataSourceConfig();
};

IModelConfig *getModelConfig() { return new IModelConfig(); }

IServerConfig *getServerConfig() { return nullptr; }

}; // namespace KRAI

#endif // BENCHMARK_CONFIG
