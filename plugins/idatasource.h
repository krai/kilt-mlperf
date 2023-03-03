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

#ifndef IDATA_SOURCE_H
#define IDATA_SOURCE_H

#include <vector>
#include <mutex>
#include <thread>

#include "iconfig.h"

namespace KRAI {

//----------------------------------------------------------------------

class IDataSource {
public:
  IDataSource(std::vector<int> &affinities) {

    CPU_ZERO(&cpu_affinity);
    for (int a = 0; a < affinities.size(); ++a) {
      CPU_SET(affinities[a], &cpu_affinity);
    }
  }

  virtual ~IDataSource() {}

  virtual void *getSamplePtr(int sample_idx, int buffer_idx) = 0;

  virtual const int getNumAvailableSampleFiles() = 0;

  virtual const int getNumMaxSamplesInMemory() = 0;

  virtual void loadSamplesImpl(void *) = 0;

  virtual void unloadSamples(void *user) = 0;

  void loadSamples(void *user) {

    mtx_load_samples.lock();

    std::thread t(&IDataSource::loadSamplesMutex, this, user);

    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpu_affinity);

    mtx_load_samples.unlock();

    t.join();
  }

private:
  void loadSamplesMutex(void *user) {

    mtx_load_samples.lock();
    loadSamplesImpl(user);
    mtx_load_samples.unlock();
  }

  cpu_set_t cpu_affinity;
  std::mutex mtx_load_samples;
};

IDataSource *dataSourceConstruct(IConfig *config, std::vector<int> affinities);

} // namespace KRAI

#endif // IDATA_SOURCE_H
