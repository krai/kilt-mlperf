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
