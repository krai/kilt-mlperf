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

#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#pragma once

#include <fstream>
#include <sstream>

#include "config_tools.h"
#include "iconfig.h"

namespace KRAI {

//----------------------------------------------------------------------

class ClassificationDataSourceConfig : public IDataSourceConfig {

public:
  virtual const int getImageSize() const {
    return image_size;
  };
  virtual const int getNumChannels() const {
    return num_channels;
  };
  virtual const bool getHasBackgroundClass() const {
    return has_background_class;
  };

  virtual const std::vector<std::string> &getListOfImageFilenames() const {
    return _available_image_list;
  };
  virtual const std::string &getDatasetDir() const {
    return images_dir;
  };

  virtual const int getMaxImagesInMemory() const {
    return images_in_memory_max;
  };

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
      getenv_i("CK_ENV_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE");

  const int num_channels = 3;

  const bool has_background_class =
      getenv_s("ML_MODEL_HAS_BACKGROUND_CLASS") == "YES";

  const std::string available_images_file =
      getenv_s("CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF");

  const std::string images_dir =
      getenv_s("CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR");

  const int images_in_memory_max = getenv_i("CK_LOADGEN_BUFFER_SIZE");

  std::vector<std::string> _available_image_list;
};

IDataSourceConfig *getDataSourceConfig() {
  return new ClassificationDataSourceConfig();
};

class ModelConfig : public IModelConfig {
public:
  virtual ~ModelConfig() {};
};

IModelConfig *getModelConfig() { return new ModelConfig(); }

}; // namespace KRAI

#endif // BENCHMARK_CONFIG
