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

#ifndef HARNESS_CONFIG_H
#define HARNESS_CONFIG_H

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <assert.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>

#include "config_tools.h"
#include "iconfig.h"

namespace KRAI {

class HarnessConfig {

public:
  const int getVerbosity() const {
    return verbosity_level;
  };
  const bool triggerColdRun() const {
    return trigger_cold_run;
  };
  const std::string getMLPerfConfigPath() const {
    return mlperf_conf_path;
  };
  const std::string getUserConfPath() const {
    return user_conf_path;
  };
  const std::string getModelName() const {
    return model_name;
  };
  const std::string getLoadgenScenario() const {
    return scenario_string;
  };
  const std::string getLoadgenMode() const {
    return mode_string;
  };

private:
  const bool trigger_cold_run = getenv_b("CK_LOADGEN_TRIGGER_COLD_RUN");
  const int verbosity_level = getenv_i("CK_VERBOSE");

  const std::string mlperf_conf_path =
      getenv_s("CK_ENV_MLPERF_INFERENCE_MLPERF_CONF");
  const std::string user_conf_path = getenv_s("CK_LOADGEN_USER_CONF");
  const std::string model_name =
      getenv_opt_s("ML_MODEL_MODEL_NAME", "unknown_model");
  const std::string scenario_string = getenv_s("CK_LOADGEN_SCENARIO");
  const std::string mode_string = getenv_s("CK_LOADGEN_MODE");
};

}; // namespace KRAI
#endif
