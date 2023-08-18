//
// MIT License
//
// Copyright (c) 2023 Krai Ltd
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

#ifndef DEVICE_CONFIG_H
#define DEVICE_CONFIG_H

#include "config/config_tools/config_tools.h"
#include "iconfig.h"

namespace KRAI {

class TensorRTDeviceConfig : public IDeviceConfig {

public:
  virtual const std::string getModelRoot() const { return model_root; }
  virtual const int getNumberOfStreams() const { return number_of_streams; }
  virtual const int getMaxSeqLen() const { return max_seq_len; }
  virtual const int getMaxBatchSize() const { return max_batch_size; }
  virtual const std::string getInputMemoryManagementStrategyName() const {
    return input_memory_management_strategy_name;
  }
  virtual const std::string getOutputMemoryManagementStrategyName() const {
    return output_memory_management_strategy_name;
  }
  virtual const std::string getPluginsPath() const { return plugins_path; }

private:
  std::string model_root =
      alter_str(getconfig_c("KILT_MODEL_ROOT"), std::string(""));
  const int number_of_streams =
      alter_str_i(getconfig_c("KILT_DEVICE_TENSORRT_NUMBER_OF_STREAMS"), 1);
  const int max_seq_len =
      alter_str_i(getconfig_c("KILT_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH"), 384);
  const int max_batch_size =
      alter_str_i(getconfig_c("KILT_DEVICE_TENSORRT_BATCH_SIZE"), 256);
  std::string input_memory_management_strategy_name =
      alter_str(getconfig_c("KILT_DEVICE_TENSORRT_INPUT_MEMORY_MANAGEMENT_STRATEGY"), std::string("DEFAULT"));
  std::string output_memory_management_strategy_name =
      alter_str(getconfig_c("KILT_DEVICE_TENSORRT_OUTPUT_MEMORY_MANAGEMENT_STRATEGY"), std::string("DEFAULT"));
  std::string plugins_path =
      alter_str(getconfig_c("KILT_DEVICE_TENSORRT_PLUGINS_PATH"), "../kilt-mlperf-dev/plugins/libnmsoptplugin.so");
};

IDeviceConfig *getDeviceConfig() { return new TensorRTDeviceConfig(); }

}; // namespace KRAI
#endif
