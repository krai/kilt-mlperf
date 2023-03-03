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
