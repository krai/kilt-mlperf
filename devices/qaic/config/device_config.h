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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.POSSIBILITY OF SUCH DAMAGE.
//


#ifndef DEVICE_CONFIG_FROM_ENV_H
#define DEVICE_CONFIG_FROM_ENV_H

#include "config_tools.h"
#include "iconfig.h"

namespace KRAI {

class QAicDeviceConfigFromEnv : public IDeviceConfig {

public:
  // Per device config
  virtual const int getActivationCount() const { return qaic_activation_count; }
  virtual const int getSetSize() const { return qaic_set_size; }
  virtual const int getNumThreadsPerQueue() const {
    return qaic_threads_per_queue;
  }
  virtual const int getInputCount() const { return qaic_input_count; }
  virtual const int getOutputCount() const { return qaic_output_count; }
  virtual const int getBatchSize() const { return qaic_batch_size; }
  virtual const int getInputSelect() const { return qaic_input_select; }
  virtual const std::string getSkipStage() const { return qaic_skip_stage; }
  virtual const std::string getModelRoot() const { return qaic_model_root; }
  virtual const bool ringfenceDeviceDriver() const {
    return qaic_ringfence_driver;
  }

  virtual const int getSamplesQueueDepth() const { return samples_queue_depth; }

  virtual const int getSchedulerYieldTime() { return scheduler_yield_time; }
  virtual const int getEnqueueYieldTime() { return enqueue_yield_time; }

private:
  const char *qaic_model_root = getenv_c("CK_ENV_QAIC_MODEL_ROOT");

  const int qaic_activation_count =
      alter_str_i(getenv_c("CK_ENV_QAIC_ACTIVATION_COUNT"), 1);

  const int qaic_set_size =
      alter_str_i(getenv_c("CK_ENV_QAIC_QUEUE_LENGTH"), 4);

  const int qaic_threads_per_queue =
      alter_str_i(getenv_c("CK_ENV_QAIC_THREADS_PER_QUEUE"), 4);

  const int qaic_input_count = getenv_i("CK_ENV_QAIC_INPUT_COUNT");

  const int qaic_output_count = getenv_i("CK_ENV_QAIC_OUTPUT_COUNT");

  const int qaic_batch_size = getenv_i("CK_ENV_QAIC_MODEL_BATCH_SIZE");

  std::string qaic_skip_stage =
      alter_str(getenv_c("CK_ENV_QAIC_SKIP_STAGE"), std::string(""));

  const int qaic_input_select =
      alter_str_i(getenv_c("CK_ENV_QAIC_INPUT_SELECT"), 0);

  const int samples_queue_depth =
      alter_str_i(getenv_c("KILT_DEVICE_QAIC_SAMPLES_QUEUE_DEPTH"), 8);

  const bool qaic_ringfence_driver =
      getenv_opt_b(std::string("KILT_DEVICE_QAIC_RINGFENCE_DRIVER"), true);

  const int scheduler_yield_time =
      alter_str_i(getenv_c("KILT_DEVICE_SCHEDULER_YIELD_TIME"), -1);

  const int enqueue_yield_time =
      alter_str_i(getenv_c("KILT_DEVICE_ENQUEUE_YIELD_TIME"), -1);
};

IDeviceConfig *getDeviceConfig() { return new QAicDeviceConfigFromEnv(); }

}; // namespace KRAI
#endif
