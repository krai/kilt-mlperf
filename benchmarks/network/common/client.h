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

#ifndef BENCHMARK_IMPL_H
#define BENCHMARK_IMPL_H

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#include "benchmarks/network/common/connection.h"
#include "benchmarks/network/common/config/network_config.h"

#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <queue>

using namespace std;
using namespace KRAI;

static int out;

class KILTClient {

public:
  KILTClient() {
    config = new IConfig();
    std::vector<int> datasource_affinity;
    datasource_affinity.push_back(0);
    data_source = dataSourceConstruct(config, datasource_affinity);

    std::cout << "Attempting to connect to " << ncc.getNetworkServerIPAddress()
              << ":" << ncc.getNetworkServerPort() << std::endl;

    payload_size = ncc.getPayloadSize();

    init_client(ncc.getNetworkServerIPAddress().c_str(),
                ncc.getNetworkServerPort());

    for (int s = 0; s < ncc.getNumSockets(); ++s) {
      conns.push_back(std::make_unique<ClientConnection>());
      std::cout << "Connected (" << s << ")" << std::endl;
    }

    receiver_conn = GetConn();

    terminate = false;
    receiver = std::thread(&KILTClient::Receiver, this);

    for (int t = 0; t < ncc.getNumSockets(); ++t) {
      transmitters.push_back(std::thread(&KILTClient::Transmitter, this));
    }
  }

  ~KILTClient() {

    terminate = true;
    receiver.detach();

    for (int t = 0; t < ncc.getNumSockets(); ++t)
      transmitters[t].detach();

    delete config;
    delete data_source;

    // tell server to detach
    auto conn = GetConn();
    uintptr_t message = -2;
    conn->write(&message, sizeof(uintptr_t));
    ReleaseConn(std::move(conn));
  }

  void Inference(const std::vector<mlperf::QuerySample> &samples) {
    samples_mtx.lock();
    for (int s = 0; s < samples.size(); ++s) {
      samples_queue.push(samples[s]);
    }
    samples_mtx.unlock();
  }

  void Transmitter() {
    mlperf::QuerySample sample;

    while (!terminate) {

      std::this_thread::yield();

      samples_mtx.lock();
      if (!samples_queue.empty()) {
        sample = samples_queue.front();
        samples_queue.pop();
        // std::cout << "Got Sample." << std::endl;
        samples_mtx.unlock();
      } else {
        samples_mtx.unlock();
        continue;
      }

      auto conn = GetConn();

      conn->write({
          {&sample.id, sizeof(uintptr_t)},
          {&sample.index, sizeof(size_t)},
          {data_source->getSamplePtr(sample.index, 0), payload_size},
      });

      if (ncc.getVerbosityLevel())
        std::cout << ">";

      ReleaseConn(std::move(conn));
    }
  }

  void LoadNextBatch(void *user) { data_source->loadSamples(user); }

  void UnloadBatch(void *user) { data_source->unloadSamples(user); }

  const int AvailableSamplesMax() {
    return data_source->getNumAvailableSampleFiles();
  }

  const int SamplesInMemoryMax() {
    return data_source->getNumMaxSamplesInMemory();
  }

  const std::string &UniqueServerID() {

    auto conn = GetConn();

    // request unique id.
    uintptr_t message = -1;

    conn->write(&message, sizeof(uintptr_t));
    char buffer[128];

    assert(conn->read(&buffer, sizeof(buffer)));

    uid = std::string(buffer);

    std::cout << "UID: " << uid << std::endl;

    ReleaseConn(std::move(conn));

    return uid;
  }

  void ColdRun() {}

  void Receiver() {

    static float response_data[65536]; // unfeasibly big number

    while (!terminate) {
      uintptr_t buffer;
      uint32_t length;
      assert(receiver_conn->read(&buffer, sizeof(uintptr_t)));
      assert(receiver_conn->read(&length, sizeof(uint32_t)));
      assert(receiver_conn->read(response_data, sizeof(float) * length));
      ++out;
      // std::cout << "Received " << buffer << " " << in << " " << out <<
      // std::endl;

      std::vector<mlperf::QuerySampleResponse> responses;

      responses.push_back(
          {buffer, uintptr_t(response_data), sizeof(float) * length});

      mlperf::QuerySamplesComplete(responses.data(), responses.size());
      if (ncc.getVerbosityLevel())
        std::cout << "<";
    }

    ReleaseConn(std::move(receiver_conn));
  }

private:
  std::unique_ptr<ClientConnection> GetConn() {
    std::unique_ptr<ClientConnection> conn = nullptr;

    while (conn == nullptr) {
      mtx_send.lock();
      if (conns.size() != 0) {
        conn = std::move(conns.back());
        conns.pop_back();
      }
      mtx_send.unlock();
    }

    return conn;
  }

  void ReleaseConn(std::unique_ptr<ClientConnection> conn) {
    mtx_send.lock();
    conns.push_back(std::move(conn));
    mtx_send.unlock();
  }

  IConfig *config;
  IDataSource *data_source;

  std::vector<std::unique_ptr<ClientConnection>> conns;
  std::unique_ptr<ClientConnection> receiver_conn;

  std::mutex mtx_send;

  std::queue<mlperf::QuerySample> samples_queue;
  std::mutex samples_mtx;

  bool terminate;
  std::thread receiver;

  std::vector<std::thread> transmitters;

  std::string uid;

  NetworkClientConfig ncc;

  std::string unique_id = "TODO GET THIS FROM THE SERVER";

  int payload_size;
};

typedef KILTClient KILT;

#endif // BENCHMARK_IMPL_H
