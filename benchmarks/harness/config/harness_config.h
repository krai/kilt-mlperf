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

#include "config/config_tools/config_tools.h"
#include "iconfig.h"

namespace KRAI {

class HarnessConfig {

public:
  const int getVerbosity() const { return verbosity_level; };
  const bool triggerColdRun() const { return trigger_cold_run; };
  const std::string getMLPerfConfigPath() const { return mlperf_conf_path; };
  const std::string getUserConfPath() const { return user_conf_path; };
  const std::string getModelName() const { return model_name; };
  const std::string getLoadgenScenario() const { return scenario_string; };
  const std::string getLoadgenMode() const { return mode_string; };

private:
  const bool trigger_cold_run = getconfig_b("LOADGEN_TRIGGER_COLD_RUN");
  const int verbosity_level = getconfig_i("KILT_VERBOSE");

  const std::string mlperf_conf_path = getconfig_s("LOADGEN_MLPERF_CONF");
  const std::string user_conf_path = getconfig_s("LOADGEN_USER_CONF");
  const std::string model_name =
      getconfig_opt_s("KILT_MODEL_NAME", "unknown_model");
  const std::string scenario_string = getconfig_s("LOADGEN_SCENARIO");
  const std::string mode_string = getconfig_s("LOADGEN_MODE");
};

}; // namespace KRAI
#endif
