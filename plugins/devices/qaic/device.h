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

#ifndef DEVICE_H
#define DEVICE_H

#include <queue>

#include "QAicInfApi.h"
#include "imodel.h"
#include "idatasource.h"
#include "idevice.h"

//#define NO_QAIC
//#define ENQUEUE_SHIM_THREADED
//#define ENQUEUE_SHIM_THREADED_COUNT 2
#define ENQUEUE_SHIM_THREADED_COUNT 0

using namespace KRAI;
using namespace qaic_api;

template <typename Sample> class Device;

template <typename Sample> struct Payload {
  std::vector<Sample> samples;
  int device;
  int activation;
  int set;
  Device<Sample> *dptr;
};

template <typename Sample> class RingBuffer {

public:
  RingBuffer(int d, int a, int s) {
    size = s;
    for (int i = 0; i < s; ++i) {
      auto p = new Payload<Sample>;
      p->set = i;
      p->activation = a;
      p->device = d;
      q.push(p);
    }
  }

  RingBuffer(int d, int a, int s, Device<Sample> *dptr) {
    size = s;
    for (int i = 0; i < s; ++i) {
      auto p = new Payload<Sample>;
      p->set = i;
      p->activation = a;
      p->device = d;
      p->dptr = dptr;
      q.push(p);
    }
  }

  virtual ~RingBuffer() {
    while (!q.empty()) {
      auto f = q.front();
      q.pop();
      delete f;
    }
  }

  Payload<Sample> *getPayload() {
    std::unique_lock<std::mutex> lock(mtx);
    if (q.empty())
      return nullptr;
    else {
      auto f = q.front();
      q.pop();
      return f;
    }
  }

  void release(Payload<Sample> *p) {
    std::unique_lock<std::mutex> lock(mtx);
    // std::cout << "Release before: " << front << " end: " << end << std::endl;
    if (q.size() == size || p == nullptr) {
      std::cerr << "extra elem in the queue" << std::endl;
      // assert(1);
    }
    q.push(p);
    // std::cout << "Release after: " << front << " end: " << end << std::endl;
  }

  void debug() {
    // std::cout << "QUEUE front: " << front << " end: " << end << std::endl;
  }

private:
  std::queue<Payload<Sample> *> q;
  int size;
  bool isEmpty;
  std::mutex mtx;
};

typedef void (*DeviceExec)(void *data);

template <typename Sample> class Device : public IDevice<Sample> {

public:
  virtual int Inference(std::vector<Sample> samples) {

    if (sback >= sfront + samples_queue_depth)
      return -1;

    samples_queue[sback % samples_queue_depth] = samples;
    ++sback;

    return samples_queue_depth - (sback - sfront);
  }

  ~Device() {
    scheduler_terminate = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    scheduler.join();

#ifdef ENQUEUE_SHIM_THREADED
    shim_terminate = true;
    for (int i = 0; i < num_setup_threads; ++i)
      payload_threads[i].join();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
#endif

#ifndef NO_QAIC
    delete runner;
#endif
  }

private:
  virtual void DeviceInit(IModel *_model, IDataSource *_data_source,
                          IDeviceConfig *_config, int hw_id,
                          std::vector<int> *aff) {

    scheduler_terminate = false;

    model = _model;
    data_source = _data_source;
    config = _config;

    samples_queue_depth = _config->getSamplesQueueDepth();

    scheduler_yield_time = _config->getSchedulerYieldTime();
    enqueue_yield_time = _config->getEnqueueYieldTime();

#ifndef NO_QAIC

    std::cout << "Creating device " << hw_id << std::endl;
    runner = new QAicInfApi();

    runner->setModelBasePath(config->getModelRoot());
    runner->setNumActivations(config->getActivationCount());
    runner->setSetSize(config->getSetSize());
    runner->setNumThreadsPerQueue(config->getNumThreadsPerQueue());
    runner->setSkipStage(config->getSkipStage());

    QStatus status = runner->init(hw_id, PostResults);

    if (status != QS_SUCCESS)
      throw "Failed to invoke qaic";

    buffers_in.resize(config->getActivationCount());
    buffers_out.resize(config->getActivationCount());

    std::cout << "Model input count: " << config->getInputCount() << std::endl;
    std::cout << "Model output count: " << config->getOutputCount()
              << std::endl;

    // get references to all the buffers
    for (int a = 0; a < config->getActivationCount(); ++a) {
      buffers_in[a].resize(config->getSetSize());
      buffers_out[a].resize(config->getSetSize());
      for (int s = 0; s < config->getSetSize(); ++s) {
        for (int i = 0; i < config->getInputCount(); ++i) {
          buffers_in[a][s].push_back((void *)runner->getBufferPtr(a, s, i));
        }
        for (int o = 0; o < config->getOutputCount(); ++o) {
          buffers_out[a][s].push_back(
              (void *)runner->getBufferPtr(a, s, o + config->getInputCount()));
        }
      }
    }

#else
    std::cout << "Creating dummy device " << hw_id << std::endl;
#endif

    // create enough ring buffers for each activation
    ring_buf.resize(config->getActivationCount());

    // populate ring buffer
    for (int a = 0; a < config->getActivationCount(); ++a)
      ring_buf[a] = new RingBuffer<Sample>(0, a, config->getSetSize(), this);

    samples_queue.resize(samples_queue_depth);
    sfront = sback = 0;

    // Kick off the scheduler
    scheduler = std::thread(&Device::QueueScheduler, this);

    // Set the affinity of the scheduler
    std::cout << "Scheduler thread " << aff->back() << std::endl;

    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    int cpu = aff->back();
    CPU_SET(cpu, &cpu_set);

    pthread_setaffinity_np(scheduler.native_handle(), sizeof(cpu_set_t),
                           &cpu_set);

    aff->pop_back();

#ifdef ENQUEUE_SHIM_THREADED
    shim_terminate = false;

    num_setup_threads = ENQUEUE_SHIM_THREADED_COUNT;

    payloads.resize(num_setup_threads, nullptr);

    for (int i = 0; i < num_setup_threads; ++i) {

      std::cout << "Shim thread " << aff->back() << std::endl;
      payload_threads.push_back(std::thread(&Device::EnqueueShim, this, i));

      // set the affinity of the shim
      cpu_set_t cpu_set;
      CPU_ZERO(&cpu_set);

      int cpu = aff->back();

      CPU_SET(cpu, &cpu_set);

      pthread_setaffinity_np(payload_threads.back().native_handle(),
                             sizeof(cpu_set_t), &cpu_set);

      aff->pop_back();
    }

#else
    payloads.resize(1, nullptr);
    shim_terminate = true;
#endif
  }

  void EnqueueShim(int id) {

    do {
      // std::cout << "Shim " << sched_getcpu() << std::endl;
      if (payloads[id] != nullptr) {
        Payload<Sample> *p = payloads[id];

#ifndef NO_QAIC
        // set the images
        if (config->getInputSelect() == 0) {
          model->configureWorkload(data_source, &(p->samples),
                                   buffers_in[p->activation][p->set]);
        } else if (config->getInputSelect() == 1) {
          assert(false);
          // TODO: We no longer support this as we're copying directly into the
          // buffer?
          // void *samples_ptr = session->getSamplePtr(p->samples[0].index);
          // runner->setBufferPtr(p->activation, p->set, 0, samples_ptr);
        } else {
          // Do nothing - random data
        }

        // std::cout << "Issuing to hardware" << std::endl;
        QStatus status = runner->run(p->activation, p->set, p);
        if (status != QS_SUCCESS)
          throw "Failed to invoke qaic";
#else
        PostResults(NULL, QAIC_EVENT_DEVICE_COMPLETE, p);
#endif

        payloads[id] = nullptr;
      } else {
        if (enqueue_yield_time)
          std::this_thread::sleep_for(
              std::chrono::microseconds(enqueue_yield_time));
      }

    } while (!shim_terminate);
  }

  void QueueScheduler() {

    // current activation index
    int activation = -1;

    std::vector<Sample> qs(config->getBatchSize());

    while (!scheduler_terminate) { // loop forever waiting for input
      // std::cout << "Scheduler " << sched_getcpu() << std::endl;
      if (sfront == sback) {
        // No samples then post last results and continue
        // Device::PostResults(nullptr, QAIC_EVENT_DEVICE_COMPLETE, this);
        if (scheduler_yield_time)
          std::this_thread::sleep_for(
              std::chrono::microseconds(scheduler_yield_time));

        continue;
      }

      // copy the image list and remove from queue
      qs = samples_queue[sfront % samples_queue_depth];
      ++sfront;

      // if(config->getVerbosityServer())
      //  std::cout << "<" << sback - sfront << ">";

      while (!scheduler_terminate) {

        activation = (activation + 1) % config->getActivationCount();

        Payload<Sample> *p = ring_buf[activation]->getPayload();

        // if no hardware slots available then increment the activation
        // count and then continue
        if (p == nullptr) {
          if (scheduler_yield_time)
            std::this_thread::sleep_for(
                std::chrono::microseconds(scheduler_yield_time));
          continue;
        }

        // add the image samples to the payload
        p->samples = qs;

#ifdef ENQUEUE_SHIM_THREADED
        int round_robin = 0;

        while (payloads[round_robin] != nullptr) {
          std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        payloads[round_robin] = p;

        // std::cout << " " << round_robin;
        round_robin = (round_robin + 1) % num_setup_threads;
#else
        // place the payload in the first slot and call shim directly
        payloads[0] = p;
        EnqueueShim(0);
#endif
        break;
      }
    }
    std::cout << "QAIC Device Scheduler terminating..." << std::endl;
  }

  static void PostResults(QAicEvent *event,
                          QAicEventCompletionType eventCompletion,
                          void *userData) {

    // Send results via queue scheduler

    if (eventCompletion == QAIC_EVENT_DEVICE_COMPLETE) {

      Payload<Sample> *p = (Payload<Sample> *)userData;

      // p->dptr->mtx_results.lock();

      // get the data from the hardware
      p->dptr->model->postprocessResults(
          &(p->samples), p->dptr->buffers_out[p->activation][p->set]);

      p->dptr->ring_buf[p->activation]->release(p);
      // p->dptr->mtx_results.unlock();
    }
  }

  cpu_set_t cpu_affinity;

  QAicInfApi *runner;

  std::vector<RingBuffer<Sample> *> ring_buf;

  std::vector<std::vector<Sample> > samples_queue;
  std::atomic<int> sfront, sback;
  int samples_queue_depth;

  std::mutex mtx_queue;
  std::mutex mtx_ringbuf;

  int num_setup_threads;
  std::vector<Payload<Sample> *> payloads;
  std::vector<std::thread> payload_threads;

  std::thread scheduler;
  bool shim_terminate;
  std::atomic<bool> scheduler_terminate;

  IDeviceConfig *config;

  std::mutex mtx_results;

  // activations, set, input buffers
  std::vector<std::vector<std::vector<void *> > > buffers_in;

  // activation, set, output buffers
  std::vector<std::vector<std::vector<void *> > > buffers_out;

  IModel *model;
  IDataSource *data_source;

  int scheduler_yield_time;
  int enqueue_yield_time;
};

template <typename Sample>
IDevice<Sample> *createDevice(IModel *_model, IDataSource *_data_source,
                              IDeviceConfig *_config, int hw_id,
                              std::vector<int> aff) {
  IDevice<Sample> *d = new Device<Sample>();
  d->Construct(_model, _data_source, _config, hw_id, aff);
  return d;
}

#endif // DEVICE_H
