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

#ifndef DEVICE_CONFIG_H
#define DEVICE_CONFIG_H

#include "config/config_tools/config_tools.h"
#include "iconfig.h"

namespace KRAI {

class SnpeDeviceConfig : public IDeviceConfig {

public:
  virtual const std::string getModelRoot() const { return snpe_model_root; }
  virtual const std::string getBackendType() const { return snpe_backend_type; }
  virtual const std::string getPerformanceProfile() const {
    return snpe_performance_profile;
  }

private:
  const std::string snpe_model_root = getconfig_s("KILT_MODEL_ROOT");
  const std::string snpe_backend_type = getconfig_s("KILT_BACKEND_TYPE");
  const std::string snpe_performance_profile =
      getconfig_s("SNPE_PERFORMANCE_PROFILE");
};

IDeviceConfig *getDeviceConfig() { return new SnpeDeviceConfig(); }

}; // namespace KRAI
#endif
