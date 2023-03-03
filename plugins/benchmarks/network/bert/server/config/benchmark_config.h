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

//----------------------------------------------------------------------

class SquadDataSourceConfig : public IDataSourceConfig {

public:
  const int getDataSourceSequenceLength() {
    return max_seq_length;
  };

private:
  const int max_seq_length =
      getenv_i("CK_ENV_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH");
};

IDataSourceConfig *getDataSourceConfig() { return new SquadDataSourceConfig(); }

class BertModelConfig : public IModelConfig {
public:
  int getModelSequenceLength() { return model_packed_seq_len; }

  const int model_packed_seq_len =
      alter_str_i(getenv_c("ML_MODEL_SEQ_LENGTH"), 384);
};

IModelConfig *getModelConfig() { return new BertModelConfig(); }

class NetworkServerConfig {
public:
  const int getNetworkServerPort() { return port; }

  const int getVerbosityLevel() { return verbosity_level; }

  const int verbosity_level = getenv_i("CK_VERBOSE");
  const int port = alter_str_i(getenv_c("NETWORK_SERVER_PORT"), 8080);
};

}; // namespace KRAI
#endif
