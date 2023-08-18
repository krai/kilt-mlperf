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

#ifndef ICONFIG_H
#define ICONFIG_H

#include "config/config_tools/config_tools.h"
#include <assert.h>
#include <sstream>

namespace KRAI {

class IServerConfig {
public:
  // Server config
  virtual const int getMaxWait() const = 0;
  virtual const int getVerbosity() const = 0;
  virtual const int getVerbosityServer() const = 0;
  virtual const int getBatchSize() const = 0;

  virtual const int getDeviceCount() const = 0;
  virtual const int getDeviceId(int idx) const = 0;
  virtual const std::vector<int> getDeviceAffinity(int device_id) = 0;

  virtual const int getDataSourceIdForDevice(int device) = 0;
  virtual const int getDataSourceCount() = 0;
  virtual const std::vector<int> getDataSourceAffinity(int data_source_id) = 0;

  virtual const std::string &getUniqueServerID() = 0;

  virtual const int getSchedulerYieldTime() = 0;
  virtual const int getDispatchYieldTime() = 0;
};

class IDeviceConfig {
public:
};

class IModelConfig {
public:
  enum IO_TYPE {
    FLOAT16,
    FLOAT32,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    HALF
  };

  IModelConfig() {

    model_batch_size = getconfig_i("KILT_MODEL_BATCH_SIZE");
    if (model_batch_size < 1) {
      std::cerr << "Batch size set to invalid size " << model_batch_size
                << std::endl;
    }

    const char *input_format = getconfig_c("KILT_MODEL_INPUT_FORMAT");
    const char *output_format = getconfig_c("KILT_MODEL_OUTPUT_FORMAT");

    if (input_format == nullptr || output_format == nullptr)
      std::cerr << "No input/output format string found" << std::endl;

    std::string kilt_model_input_format(input_format);
    std::string kilt_model_output_format(output_format);

    // parse input
    std::stringstream ss_ids(kilt_model_input_format);
    while (ss_ids.good()) {
      // get : : bounded substring
      std::string substr;
      std::getline(ss_ids, substr, ':');
      std::stringstream ss_dse(substr);
      // parse type
      if (!ss_dse.good())
        std::cerr << "Ill formatted input format string" << std::endl;
      std::string substre;
      std::getline(ss_dse, substre, ',');
      IO_TYPE iotype = strToIOType(substre);
      model_input_types.push_back(iotype);
      // parse sizes
      std::vector<int> e;
      while (ss_dse.good()) {
        std::string substre;
        std::getline(ss_dse, substre, ',');
        int size = std::stoi(substre);
        e.push_back(size);
      }
      // try to set batch size
      if (e.at(0) != -1 && e.at(0) != model_batch_size) {
        std::cout << "Model dimension is fixed, failed to set batch size to "
                  << model_batch_size << std::endl;
      } else {
        std::cout << "Setting batch size to " << model_batch_size << std::endl;
        e[0] = model_batch_size;
      }
      model_input_dimensions.push_back(e);
    }

    // parse output
    std::stringstream ss_ods(kilt_model_output_format);
    while (ss_ods.good()) {
      // get : : bounded substring
      std::string substr;
      std::getline(ss_ods, substr, ':');
      std::stringstream ss_dse(substr);
      // parse type
      if (!ss_dse.good())
        std::cerr << "Ill formatted input format string" << std::endl;
      std::string substre;
      std::getline(ss_dse, substre, ',');
      IO_TYPE iotype = strToIOType(substre);
      model_output_types.push_back(iotype);
      // parse sizes
      std::vector<int> e;
      while (ss_dse.good()) {
        std::string substre;
        std::getline(ss_dse, substre, ',');
        int size = std::stoi(substre);
        e.push_back(size);
      }
      // try to set batch size
      if (e.at(0) != -1 && e.at(0) != model_batch_size) {
        std::cout << "Model dimension is fixed, failed to set batch size to "
                  << model_batch_size << std::endl;
      } else {
        std::cout << "Setting batch size to " << model_batch_size << std::endl;
        e[0] = model_batch_size;
      }
      model_output_dimensions.push_back(e);
    }

    model_input_count = model_input_dimensions.size();
    assert(model_input_count >= 1);
    model_output_count = model_output_dimensions.size();
    assert(model_output_count >= 1);

    for (int i = 0; i < model_input_count; i++) {
      int bufferSize = calculateBufferSize(getInputDimensions(i));
      int byteSize = bufferSize * getInputDatatypeSize(i);
      model_input_sizes.push_back(bufferSize);
      model_input_byte_sizes.push_back(byteSize);
    }

    for (int i = 0; i < model_output_count; i++) {
      int bufferSize = calculateBufferSize(getOutputDimensions(i));
      int byteSize = bufferSize * getOutputDatatypeSize(i);
      model_output_sizes.push_back(bufferSize);
      model_output_byte_sizes.push_back(byteSize);
    }
  }

  const int calculateBufferSize(std::vector<int> dimensions) {
    int accum = 1;
    for (int i : dimensions)
      accum *= i;
    return accum;
  }

  const IO_TYPE strToIOType(std::string str) const {
    if (str == "FLOAT16")
      return FLOAT16;
    else if (str == "FLOAT32")
      return FLOAT32;
    else if (str == "INT8")
      return INT8;
    else if (str == "INT16")
      return INT16;
    else if (str == "INT32")
      return INT32;
    else if (str == "INT64")
      return INT64;
    else if (str == "UINT8")
      return UINT8;
    else if (str == "UINT16")
      return UINT16;
    else if (str == "UINT32")
      return UINT32;
    else if (str == "UINT64")
      return UINT64;
    else if (str == "HALF")
      return HALF;
    else {
      std::cerr << "string doesn't correspond to type" << std::endl;
      return FLOAT32;
    }
  }

  const int sizeOfIOType(IO_TYPE type) const {
    switch (type) {
    case INT8:
    case UINT8:
      return 1;
    case FLOAT16:
    case INT16:
    case UINT16:
      return 2;
    case FLOAT32:
    case INT32:
    case UINT32:
      return 4;
    case INT64:
    case UINT64:
      return 8;
    default:
      std::cerr << "type not handled" << std::endl;
      return 8;
    }
  }

  virtual const int getInputCount() const { return model_input_count; }

  virtual const int getOutputCount() const { return model_output_count; }

  virtual const int getBatchSize() const { return model_batch_size; }

  virtual const IO_TYPE getInputDatatype(const int buf_idx) const {
    return model_input_types.at(buf_idx);
  }

  virtual const IO_TYPE getOutputDatatype(const int buf_idx) const {
    return model_output_types.at(buf_idx);
  }

  const std::vector<int> getInputDimensions(const int buf_idx) const {
    return model_input_dimensions.at(buf_idx);
  }

  const std::vector<int> getOutputDimensions(const int buf_idx) const {
    return model_output_dimensions.at(buf_idx);
  }

  const int getInputDimension(const int buf_idx, int dim) const {
    return model_input_dimensions.at(buf_idx).at(dim);
  }

  const int getOutputDimension(const int buf_idx, int dim) const {
    return model_output_dimensions.at(buf_idx).at(dim);
  }

  const int getInputSize(const int buf_idx) const {
    return model_input_sizes.at(buf_idx);
  }

  const int getOutputSize(const int buf_idx) const {
    return model_output_sizes.at(buf_idx);
  }

  const int getInputDatatypeSize(const int buf_idx) const {
    return sizeOfIOType(getInputDatatype(buf_idx));
  }

  const int getOutputDatatypeSize(const int buf_idx) const {
    return sizeOfIOType(getOutputDatatype(buf_idx));
  }

  const int getInputByteSize(const int buf_idx) const {
    return model_input_byte_sizes.at(buf_idx);
  }

  const int getOutputByteSize(const int buf_idx) const {
    return model_output_byte_sizes.at(buf_idx);
  }

private:
  int model_input_count;
  int model_output_count;
  int model_batch_size;

  std::vector<std::vector<int>> model_input_dimensions;
  std::vector<std::vector<int>> model_output_dimensions;

  std::vector<IO_TYPE> model_input_types;
  std::vector<IO_TYPE> model_output_types;

  std::vector<int> model_input_sizes;
  std::vector<int> model_output_sizes;

  std::vector<int> model_input_byte_sizes;
  std::vector<int> model_output_byte_sizes;
};

class IDataSourceConfig {
public:
};

IServerConfig *getServerConfig();
IDeviceConfig *getDeviceConfig();
IModelConfig *getModelConfig();
IDataSourceConfig *getDataSourceConfig();

class IConfig {
public:
  IConfig() {
    server_cfg = getServerConfig();
    device_cfg = getDeviceConfig();
    model_cfg = getModelConfig();
    datasource_cfg = getDataSourceConfig();
  }

  IServerConfig *server_cfg;
  IDeviceConfig *device_cfg;
  IModelConfig *model_cfg;
  IDataSourceConfig *datasource_cfg;
};

}; // namespace KRAI

#endif // ICONFIG_H
