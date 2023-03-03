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

#include "config_json_tools.h"
#include "iconfig.h"

namespace KRAI {

template <char delimiter> class WordDelimitedBy : public std::string {};

template <char delimiter>
std::istream &operator>>(std::istream &is, WordDelimitedBy<delimiter> &output) {
  std::getline(is, output, delimiter);
  return is;
}

//----------------------------------------------------------------------

class ObjectDetectionDataSourceConfig : public IDataSourceConfig {

public:
  virtual const int getImageSize() const {
    return image_size;
  };
  virtual const int getNumChannels() const {
    return num_channels;
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

  ObjectDetectionDataSourceConfig() {

    assert(image_size_height == image_size_width);
    image_size = image_size_height;

    // Load list of images to be processed
    std::ifstream file(images_dir + "/" + available_images_file);
    if (!file)
      throw "Unable to open image list file " + available_images_file;

    for (std::string s; !getline(file, s).fail();) {
      std::istringstream iss(s);
      std::vector<std::string> row(
          (std::istream_iterator<WordDelimitedBy<';'> >(iss)),
          std::istream_iterator<WordDelimitedBy<';'> >());
      _available_image_list.emplace_back(row[0]);
    }
  }

private:
  const int image_size_height = getenv_i("ML_MODEL_IMAGE_HEIGHT");
  const int image_size_width = getenv_i("ML_MODEL_IMAGE_WIDTH");
  int image_size;

  const int num_channels = alter_str_i(getenv_c("ML_MODEL_IMAGE_CHANNELS"), 3);

  const std::string images_dir =
      getenv_s("CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR");

  const std::string available_images_file =
      getenv_s("CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF");

  const int images_in_memory_max = getenv_i("CK_LOADGEN_BUFFER_SIZE");

  const int qaic_batch_size = getenv_i("CK_ENV_QAIC_MODEL_BATCH_SIZE");

  std::vector<std::string> _available_image_list;
};

IDataSourceConfig *getDataSourceConfig() {
  return new ObjectDetectionDataSourceConfig();
};

class ModelConfig : public IModelConfig {
public:
  virtual ~ModelConfig() {};

  const int getMaxDetections() { return max_detections; }
  const bool disableNMS() { return disable_nms; }
  const std::string getPriorsBinPath() { return priors_bin_path; }

private:
  const std::string priors_bin_path = getenv_s("PRIOR_BIN_PATH");
  const int max_detections = alter_str_i(
      getenv_c("MAX_DETECTIONS"), getenv_c("CK_ENV_QAIC_MODEL_MAX_DETECTIONS"));
  const bool disable_nms = (getenv_c("CK_ENV_DISABLE_NMS") != NULL);
};

IModelConfig *getModelConfig() { return new ModelConfig(); }

}; // namespace KRAI

#endif // BENCHMARK_CONFIG
