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

#ifndef ICONFIG_H
#define ICONFIG_H

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
  // Per device config
  virtual const int getActivationCount() const = 0;
  virtual const int getSetSize() const = 0;
  virtual const int getNumThreadsPerQueue() const = 0;
  virtual const int getInputCount() const = 0;
  virtual const int getOutputCount() const = 0;
  virtual const int getBatchSize() const = 0;
  virtual const int getInputSelect() const = 0;
  virtual const std::string getSkipStage() const = 0;
  virtual const std::string getModelRoot() const = 0;
  virtual const int getSamplesQueueDepth() const = 0;
  virtual const bool ringfenceDeviceDriver() const = 0;

  virtual const int getSchedulerYieldTime() = 0;
  virtual const int getEnqueueYieldTime() = 0;
};

class IModelConfig {
public:
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
