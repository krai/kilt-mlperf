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

#ifndef SERVER_CONFIG_FROM_ENV_H
#define SERVER_CONFIG_FROM_ENV_H

#include "config_tools.h"
#include "iconfig.h"
#include "affinity.h"

namespace KRAI {

class ServerConfigFromEnv : public IServerConfig {

public:
  // Server settings
  virtual const int getMaxWait() const { return max_wait; }
  virtual const int getVerbosity() const { return verbosity_level; }
  virtual const int getVerbosityServer() const { return verbosity_server; }
  virtual const int getBatchSize() const { return qaic_batch_size; }

  virtual const int getDeviceCount() const { return qaic_device_count; }
  virtual const int getDeviceId(int idx) const { return qaic_hw_ids[idx]; }
  virtual const std::vector<int> getDeviceAffinity(int device_id) {
    return qaic_hw_affinities[device_id];
  }

  virtual const int getDataSourceIdForDevice(int device) {
    return qaic_hw_datasource_for_device[device];
  }
  virtual const int getDataSourceCount() { return qaic_datasource_count; }
  virtual const std::vector<int> getDataSourceAffinity(int data_source_id) {
    return qaic_datasource_affinities[data_source_id];
  }

  virtual const std::string &getUniqueServerID() { return unique_server_id; }

  virtual const int getSchedulerYieldTime() { return scheduler_yield_time; }

  virtual const int getDispatchYieldTime() { return dispatch_yield_time; }

  ServerConfigFromEnv() {

    std::stringstream ss_ids(qaic_hw_ids_str);
    while (ss_ids.good()) {
      std::string substr;
      std::getline(ss_ids, substr, ',');
      qaic_hw_ids.push_back(std::stoi(substr));
    }

    qaic_device_count = qaic_hw_ids.size();

    // calculate max range of devices
    int max_device_id = 1;
    for (int d = 0; d < qaic_hw_ids.size(); ++d) {
      if (max_device_id < (qaic_hw_ids[d] + 1))
        max_device_id = (qaic_hw_ids[d] + 1);
    }

    // fall back to calculated datasource config if none supplied
    if (qaic_datasource_config_str == "") {

      // generate datasource affinities
      for (int d = 0; d < max_device_id; ++d) {
        std::vector<int> affinity;
        affinity.push_back(AFFINITY_CARD(d));
        qaic_datasource_affinities.push_back(affinity);
      }

      // datasource count
      qaic_datasource_count = max_device_id;

    } else {

      // config for each datasource is separated by a colon
      // - the config for a specific datasource is separated by commas
      // - the elements in the config refer to the cpu core affinity

      std::stringstream ss_ds(qaic_datasource_config_str);
      while (ss_ds.good()) {
        std::string substr;
        std::getline(ss_ds, substr, ':');

        std::vector<int> e;
        std::stringstream ss_dse(substr);
        while (ss_dse.good()) {
          std::string substre;
          std::getline(ss_dse, substre, ',');
          e.push_back(std::stoi(substre));
        }

        qaic_datasource_affinities.push_back(e);
      }

      qaic_datasource_count = qaic_datasource_affinities.size();
    }

    // fall back to calculated device config if none supplied
    if (qaic_hw_config_str == "") {

      // generate device affinities
      for (int d = 0; d < max_device_id; ++d) {
        qaic_hw_datasource_for_device.push_back(d);

        std::vector<int> affinity;
        int aff_base = AFFINITY_CARD(d);
        for (int j = 0; j < 4; j++) {
          affinity.push_back(aff_base + j);
        }
        qaic_hw_affinities.push_back(affinity);
      }
    } else {

      // config for each device is separated by a colon
      // - the config for a specific device is separated by commas
      // - the first element is an index into the datasource buffer
      // - the remaining elements is the cpu core affinity

      std::stringstream ss_dcfg(qaic_hw_config_str);
      while (ss_dcfg.good()) {
        std::string substr;
        std::getline(ss_dcfg, substr, ':');

        std::vector<int> e;
        std::stringstream ss_dcfge(substr);

        if (ss_dcfge.good()) {
          std::string substre;
          std::getline(ss_dcfge, substre, ',');
          qaic_hw_datasource_for_device.push_back(std::stoi(substre));
        }
        while (ss_dcfge.good()) {
          std::string substre;
          std::getline(ss_dcfge, substre, ',');
          e.push_back(std::stoi(substre));
        }

        qaic_hw_affinities.push_back(e);
      }
    }
  }

private:
  const int verbosity_level = getenv_i("CK_VERBOSE");

  const int verbosity_server = alter_str_i(getenv_c("CK_VERBOSE_SERVER"), 0);

  const int qaic_batch_size = getenv_i("CK_ENV_QAIC_MODEL_BATCH_SIZE");

  const int max_wait =
      alter_str_i(getenv_c("CK_ENV_QAIC_MAX_WAIT_ABS"), 100000);

  const int scheduler_yield_time =
      alter_str_i(getenv_c("KILT_SCHEDULER_YIELD_TIME"), 10);

  const int dispatch_yield_time =
      alter_str_i(getenv_c("KILT_DISPATCH_YIELD_TIME"), -1);

  //   // choice of hardware
  std::string qaic_hw_ids_str =
      alter_str(getenv_c("CK_ENV_QAIC_DEVICE_IDS"), std::string("0"));

  std::string qaic_hw_config_str =
      alter_str(getenv_c("CK_ENV_QAIC_DEVICE_CONFIG"), std::string(""));

  std::string qaic_datasource_config_str =
      alter_str(getenv_c("CK_ENV_QAIC_DATASOURCE_CONFIG"), std::string(""));

  std::string unique_server_id = alter_str(getenv_c("CK_ENV_UNIQUE_SERVER_ID"),
                                           std::string("KILT_SERVER"));

  std::vector<int> qaic_hw_ids;

  int qaic_device_count;
  int qaic_datasource_count;

  std::vector<std::vector<int> > qaic_hw_affinities;
  std::vector<std::vector<int> > qaic_datasource_affinities;
  std::vector<int> qaic_hw_datasource_for_device;
};

IServerConfig *getServerConfig() { return new ServerConfigFromEnv(); }

}; // namespace KRAI
#endif
