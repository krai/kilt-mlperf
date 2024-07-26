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

#ifndef SERVER_CONFIG_H
#define SERVER_CONFIG_H

#include <sstream>

#include "config/config_tools/config_tools.h"

#include "affinity.h"
#include "iconfig.h"

namespace KRAI {

class ServerConfig : public IServerConfig {

public:
  // Server settings
  virtual const int getMaxWait() const { return max_wait; }
  virtual const int getVerbosity() const { return verbosity_level; }
  virtual const int getVerbosityServer() const { return verbosity_server; }
  virtual const int getBatchSize() const { return batch_size; }

  virtual const int getDeviceCount() const { return device_count; }
  virtual const int getDeviceId(int idx) const { return hw_ids[idx]; }
  virtual const std::vector<int> getDeviceAffinity(int device_id) {
    return hw_affinities[device_id];
  }

  virtual const int getDataSourceIdForDevice(int device) {
    return hw_datasource_for_device[device];
  }
  virtual const int getDataSourceCount() { return datasource_count; }
  virtual const std::vector<int> getDataSourceAffinity(int data_source_id) {
    return datasource_affinities[data_source_id];
  }

  virtual const std::string &getUniqueServerID() { return unique_server_id; }

  virtual const int getSchedulerYieldTime() { return scheduler_yield_time; }

  virtual const int getDispatchYieldTime() { return dispatch_yield_time; }

  ServerConfig() {

    std::stringstream ss_ids(hw_ids_str);
    while (ss_ids.good()) {
      std::string substr;
      std::getline(ss_ids, substr, ',');
      hw_ids.push_back(std::stoi(substr));
    }

    device_count = hw_ids.size();

    // calculate max range of devices
    int max_device_id = 1;
    for (int d = 0; d < hw_ids.size(); ++d) {
      if (max_device_id < (hw_ids[d] + 1))
        max_device_id = (hw_ids[d] + 1);
    }

    // fall back to calculated datasource config if none supplied
    if (datasource_config_str == "") {

      // generate datasource affinities
      for (int d = 0; d < max_device_id; ++d) {
        std::vector<int> affinity;
        affinity.push_back(AFFINITY_CARD(d));
        datasource_affinities.push_back(affinity);
      }

      // datasource count
      datasource_count = max_device_id;

    } else {

      // config for each datasource is separated by a colon
      // - the config for a specific datasource is separated by commas
      // - the elements in the config refer to the cpu core affinity

      std::stringstream ss_ds(datasource_config_str);
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

        datasource_affinities.push_back(e);
      }

      datasource_count = datasource_affinities.size();
    }

    // fall back to calculated device config if none supplied
    if (hw_config_str == "") {

      // generate device affinities
      for (int d = 0; d < max_device_id; ++d) {
        hw_datasource_for_device.push_back(d);

        std::vector<int> affinity;
        int aff_base = AFFINITY_CARD(d);
        for (int j = 0; j < 4; j++) {
          affinity.push_back(aff_base + j);
        }
        hw_affinities.push_back(affinity);
      }
    } else {

      // config for each device is separated by a colon
      // - the config for a specific device is separated by commas
      // - the first element is an index into the datasource buffer
      // - the remaining elements is the cpu core affinity

      std::stringstream ss_dcfg(hw_config_str);
      while (ss_dcfg.good()) {
        std::string substr;
        std::getline(ss_dcfg, substr, ':');

        std::vector<int> e;
        std::stringstream ss_dcfge(substr);

        if (ss_dcfge.good()) {
          std::string substre;
          std::getline(ss_dcfge, substre, ',');
          hw_datasource_for_device.push_back(std::stoi(substre));
        }
        while (ss_dcfge.good()) {
          std::string substre;
          std::getline(ss_dcfge, substre, ',');
          e.push_back(std::stoi(substre));
        }

        hw_affinities.push_back(e);
      }
    }
  }

private:
  const int verbosity_level = getconfig_i("KILT_VERBOSE");

  const int verbosity_server =
      alter_str_i(getconfig_c("KILT_VERBOSE_SERVER"), 0);

  const int batch_size = getconfig_i("KILT_MODEL_BATCH_SIZE");

  const int max_wait = alter_str_i(getconfig_c("KILT_MAX_WAIT_ABS"), 100000);

  const int scheduler_yield_time =
      alter_str_i(getconfig_c("KILT_SCHEDULER_YIELD_TIME"), 10);

  const int dispatch_yield_time =
      alter_str_i(getconfig_c("KILT_DISPATCH_YIELD_TIME"), -1);

  //   // choice of hardware
  std::string hw_ids_str =
      alter_str(getconfig_c("KILT_DEVICE_IDS"), std::string("0"));

  std::string hw_config_str =
      alter_str(getconfig_c("KILT_DEVICE_CONFIG"), std::string(""));

  std::string datasource_config_str =
      alter_str(getconfig_c("KILT_DATASOURCE_CONFIG"), std::string(""));

  std::string unique_server_id = alter_str(
      getconfig_c("KILT_NETWORK_UNIQUE_SERVER_ID"), std::string("KILT_SERVER"));

  std::vector<int> hw_ids;

  int device_count;
  int datasource_count;

  std::vector<std::vector<int>> hw_affinities;
  std::vector<std::vector<int>> datasource_affinities;
  std::vector<int> hw_datasource_for_device;
};

IServerConfig *getServerConfig() { return new ServerConfig(); }

}; // namespace KRAI
#endif
