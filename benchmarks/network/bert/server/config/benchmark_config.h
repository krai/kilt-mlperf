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

//----------------------------------------------------------------------

class SquadDataSourceConfig : public IDataSourceConfig {

public:
  const int getDataSourceSequenceLength() { return max_seq_length; };

private:
  const int max_seq_length =
      getconfig_i("KILT_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH");
};

IDataSourceConfig *getDataSourceConfig() { return new SquadDataSourceConfig(); }

class BertModelConfig : public IModelConfig {

public:
  virtual const IO_TYPE getInputDatatype(const int buf_idx) const {
    if (strcmp(qaic_skip_stage.c_str(), "convert") == 0) {
      return UINT32;
    } else {
      return UINT64;
    }
  }

  virtual const IO_TYPE getOutputDatatype(const int buf_idx) const {
    return INT32; // never used
  }

  int getModelSequenceLength() const { return model_packed_seq_len; }

private:
  std::string qaic_skip_stage =
      alter_str(getconfig_c("KILT_DEVICE_QAIC_SKIP_STAGE"), std::string(""));

  const int model_packed_seq_len =
      alter_str_i(getconfig_c("KILT_MODEL_BERT_SEQ_LENGTH"), 384);
};

IModelConfig *getModelConfig() { return new BertModelConfig(); }

class NetworkServerConfig {
public:
  const int getNetworkServerPort() { return port; }

  const int getVerbosityLevel() { return verbosity_level; }

  const int verbosity_level = getconfig_i("KILT_VERBOSE");
  const int port = alter_str_i(getconfig_c("KILT_NETWORK_SERVER_PORT"), 8080);
};

}; // namespace KRAI
#endif
