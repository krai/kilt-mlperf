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
