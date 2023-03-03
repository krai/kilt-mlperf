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
