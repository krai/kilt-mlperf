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

#ifndef KRAI_INFERENCE_LIBRARY_H
#define KRAI_INFERENCE_LIBRARY_H

#include "iconfig.h"
#include "idevice.h"
#include "imodel.h"

#include "config/kilt_config.h"

#include <atomic>

using namespace KRAI;

template <typename Sample> class KraiInferenceLibrary {
public:
  KraiInferenceLibrary() {

    config = new IConfig();

    scheduler_yield_time = config->server_cfg->getSchedulerYieldTime();
    dispatch_yield_time = config->server_cfg->getDispatchYieldTime();

    terminate = false;

    scheduler = std::thread(&KraiInferenceLibrary::Scheduler, this);

    model = modelConstruct(config);

    for (int ds = 0; ds < config->server_cfg->getDataSourceCount(); ++ds) {
      const std::vector<int> datasource_affinity =
          config->server_cfg->getDataSourceAffinity(ds);

      std::cout << "DataSource: [" << ds << "] affinity: ";
      for (int i = 0; i < datasource_affinity.size(); ++i)
        std::cout << datasource_affinity[i] << " ";
      std::cout << std::endl;

      data_sources.push_back(dataSourceConstruct(config, datasource_affinity));
    }

    for (int dv = 0; dv < config->server_cfg->getDeviceCount(); ++dv) {

      unsigned int device_id = config->server_cfg->getDeviceId(dv);
      std::vector<int> device_affinity =
          config->server_cfg->getDeviceAffinity(device_id);
      unsigned int data_source_id =
          config->server_cfg->getDataSourceIdForDevice(device_id);

      std::cout << "Device: [" << dv << "] (data source " << data_source_id
                << ") affinity: ";
      for (int i = 0; i < device_affinity.size(); ++i)
        std::cout << device_affinity[i] << " ";
      std::cout << std::endl;

      IDevice<Sample> *device =
          createDevice<Sample>(model, data_sources[data_source_id], config,
                               device_id, device_affinity);

      devices.push_back(device);
    }

    queue_len = std::vector<uint64_t>(config->server_cfg->getDeviceCount(), 0);

    // diagnostics
    batch_trace = std::vector<uint64_t>(config->server_cfg->getBatchSize(), 0);
    distribution =
        std::vector<uint64_t>(config->server_cfg->getDeviceCount(), 0);
  }

  ~KraiInferenceLibrary() {

    terminate = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    scheduler.join();

    for (int d = 0; d < config->server_cfg->getDeviceCount(); ++d) {
      delete devices[d];
    }
    for (int ds = 0; ds < data_sources.size(); ++ds) {
      delete data_sources[ds];
    }

    std::cout << "Batch sizes dispatched: ";
    for (int t = 0; t < batch_trace.size(); ++t)
      std::cout << batch_trace[t] << " ";
    std::cout << std::endl;

    delete model;
  }

  void ColdRun() {

#if 0
    auto vl = config->server_cfg->verbosity_level;

    if (vl > 1) {
      std::cout << "Triggering a Cold Run..." << std::endl;
    } else if (vl) {
      std::cout << 'C' << std::flush;
    }

    // QStatus status = runner->run(totalSetsCompleted, totalInferencesCompleted);
    // if (status != QS_SUCCESS)
    //  throw "Failed to invoke qaic";
#endif
  }

  static void DispatchImpl(void *handle, const void *samples) {

    KraiInferenceLibrary<Sample> *ths =
        reinterpret_cast<KraiInferenceLibrary<Sample> *>(handle);

    const std::vector<Sample> *s =
        reinterpret_cast<const std::vector<Sample> *>(samples);

    ths->Dispatch(*s);
  }

  void Inference(const std::vector<Sample> &samples) {

    int num_samples = samples.size();

    for (int s = 0; s < num_samples; ++s) {

      mtx_samples_queue.lock();

      samples_queue.emplace_back(samples[s]);

      if (samples_queue.size() == config->server_cfg->getBatchSize()) {

        ++batch_trace[samples_queue.size() - 1];

        model->preprocessSamples(data_sources[0], &samples_queue, this,
                                 DispatchImpl);

        // Dispatch(samples_queue);
        samples_queue.clear();
        prev = std::chrono::steady_clock::now();
      }
      mtx_samples_queue.unlock();
    }
  }

  void LoadNextBatch(void *user) {

    auto vl = config->server_cfg->getVerbosity();

    if (vl) {
      std::cout << 'L' << std::flush;
    }

#ifndef NO_QAIC
    for (int d = 0; d < data_sources.size(); ++d) {
      data_sources[d]->loadSamples(user);
    }
#endif

    if (vl) {
      std::cout << std::endl;
    }
  }

  void UnloadBatch(void *user) {

    auto vl = config->server_cfg->getVerbosity();

    if (vl) {
      std::cout << 'U' << std::flush;
    }

#ifndef NO_QAIC
    for (int d = 0; d < data_sources.size(); ++d) {
      data_sources[d]->unloadSamples(user);
    }
#endif

    if (vl) {
      std::cout << std::endl;
    }
  }

  const int AvailableSamplesMax() {
    return data_sources[0]->getNumAvailableSampleFiles();
  }

  const int SamplesInMemoryMax() {
    return data_sources[0]->getNumMaxSamplesInMemory();
  }

  const std::string &UniqueServerID() {
    return config->server_cfg->getUniqueServerID();
  }

private:
  void Dispatch(const std::vector<Sample> &samples) {

    static int round_robin = 0;

    int done;

    while (1) {
      done = devices[round_robin]->Inference(samples);
      queue_len[round_robin] = done;
      round_robin = (round_robin + 1) % config->server_cfg->getDeviceCount();

      if (done >= 0)
        break;

      if (dispatch_yield_time)
        std::this_thread::sleep_for(
            std::chrono::microseconds(dispatch_yield_time));
    }

    ++distribution[round_robin];

#if 0
    static int counter = 0;

    if(++counter == 1000) {
    counter = 0;
    for( int x=0 ; x<config->server_cfg->getDeviceCount() ; ++x) {
      //std::cout << std::setw(2) << queue_len[x] << " ";
      std::cout << queue_len[x] << " ";
    }
    std::cout << "[ ";
    for( int x=0 ; x<config->server_cfg->getDeviceCount() ; ++x) {
      std::cout << distribution[x] << " ";
    }
    std::cout << "][ ";
    for(int t = 0; t<batch_trace.size() ; ++t)
      std::cout << batch_trace[t] << " ";
    std::cout << "]" << std::endl;
    }
#endif
  }

  void Scheduler() {

    prev = std::chrono::steady_clock::now();
    std::chrono::microseconds max_wait =
        std::chrono::microseconds(config->server_cfg->getMaxWait());

    std::cout << "MaxWait: " << config->server_cfg->getMaxWait() << std::endl;

    while (!terminate) {
      auto now = std::chrono::steady_clock::now();
      mtx_samples_queue.lock();
      int qlen = samples_queue.size();

      if (qlen) {
        if ((now - prev) > max_wait) {
          if (config->server_cfg->getVerbosityServer())
            std::cout << "(" << qlen << ")";

          ++batch_trace[samples_queue.size() - 1];
          // std::cout << "Timeout triggered." << std::endl;
          model->preprocessSamples(data_sources[0], &samples_queue, this,
                                   DispatchImpl);
          // Dispatch(samples_queue);
          samples_queue.clear();
          prev = now;
        }
      } else {
        prev = now;
      }
      mtx_samples_queue.unlock();
      std::this_thread::sleep_for(
          std::chrono::microseconds(scheduler_yield_time));
    }
    std::cout << "KILT Scheduler terminating..." << std::endl;
  }

  IConfig *config;

  std::vector<uint64_t> batch_trace;
  std::vector<uint64_t> queue_len;
  std::vector<uint64_t> distribution;

  std::vector<IDevice<Sample> *> devices;
  std::vector<IDataSource *> data_sources;

  IModel *model;

  std::vector<Sample> samples_queue;
  std::mutex mtx_samples_queue;
  std::chrono::time_point<std::chrono::steady_clock> prev;

  std::atomic<bool> terminate;
  std::thread scheduler;

  int scheduler_yield_time;
  int dispatch_yield_time;
};

#endif // KRAI_INFERENCE_LIBRARY_H
