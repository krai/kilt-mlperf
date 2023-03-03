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

#ifndef IDEVICE_H
#define IDEVICE_H

#include <iostream>

#include "imodel.h"
#include "idatasource.h"

using namespace KRAI;

template <typename Sample> class IDevice {

public:
  void Construct(IModel *_model, IDataSource *_data_source,
                 IDeviceConfig *_config, int hw_id, std::vector<int> aff) {

    cpu_set_t cpu_affinity;

    std::vector<int> aff_cpy = aff;
    int device_threads = 0;

    if (_config->ringfenceDeviceDriver()) {
      device_threads = 1;
    }

    std::cout << "Driver threads: ";
    CPU_ZERO(&cpu_affinity);
    for (int a = aff.size(); a > device_threads; a--) {
      CPU_SET(aff.back(), &cpu_affinity);
      std::cout << " " << aff.back();
      aff.pop_back();
    }
    std::cout << std::endl;

    if (!_config->ringfenceDeviceDriver()) {
      aff = aff_cpy;
    }

    mtx_init.lock();
    std::thread tin(&IDevice::DeviceInitMutex, this, _model, _data_source,
                    _config, hw_id, &aff);
    pthread_setaffinity_np(tin.native_handle(), sizeof(cpu_set_t),
                           &cpu_affinity);
    mtx_init.unlock();

    tin.join();
  }

  virtual void DeviceInit(IModel *_model, IDataSource *_data_source,
                          IDeviceConfig *_config, int hw_id,
                          std::vector<int> *aff) = 0;

  virtual int Inference(std::vector<Sample> samples) = 0;

  virtual ~IDevice() {};

private:
  virtual void DeviceInitMutex(IModel *_model, IDataSource *_data_source,
                               IDeviceConfig *_config, int hw_id,
                               std::vector<int> *aff) {

    mtx_init.lock();
    DeviceInit(_model, _data_source, _config, hw_id, aff);
    mtx_init.unlock();
  }

  std::mutex mtx_init;
};

template <typename Sample>
IDevice<Sample> *createDevice(IModel *_model, IDataSource *_data_source,
                              IDeviceConfig *_config, int hw_id,
                              std::vector<int> aff);

#endif // IDEVICE_H
