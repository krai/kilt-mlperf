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

#ifndef BENCHMARK_IMPL_H
#define BENCHMARK_IMPL_H

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#include "datasource.h"
#include "benchmark_config.h"

#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;
using namespace KRAI;

static int out;
static int in;

bool read_socket(int sock, void *buf, size_t len) {

  uint8_t *byte_buf = reinterpret_cast<uint8_t *>(buf);

  while (len != 0) {

    size_t bytes = read(sock, byte_buf, len);
    if (bytes < 0)
      return false;

    byte_buf += bytes;

    len -= bytes;
  }
  return true;
}

class KILTClient {

public:
  KILTClient() {
    config = new IConfig();
    std::vector<int> datasource_affinity;
    datasource_affinity.push_back(0);
    data_source = dataSourceConstruct(config, datasource_affinity);

    std::cout << "Attempting to connect to " << ncc.getNetworkServerIPAddress()
              << ":" << ncc.getNetworkServerPort() << std::endl;

    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      std::cout << "Could not create socket." << std::endl;
      exit(1);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(ncc.getNetworkServerPort());

    if (inet_pton(AF_INET, ncc.getNetworkServerIPAddress().c_str(),
                  &serv_addr.sin_addr) <= 0) {
      std::cout << "Address invalid / not supported." << std::endl;
      exit(1);
    }

    while ((client_fd = connect(sock, (struct sockaddr *)&serv_addr,
                                sizeof(serv_addr))) < 0) {
      std::cout << "Waiting to connect..." << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    terminate = false;
    receiver = std::thread(&KILTClient::Receiver, this);
  }

  ~KILTClient() {

    terminate = true;
    receiver.detach();

    delete config;
    delete data_source;

    // tell server to detach
    uintptr_t message = -2;
    send(sock, &message, sizeof(uintptr_t), 0);

    // closing the connected socket
    close(client_fd);
  }

  void Inference(const std::vector<mlperf::QuerySample> &samples) {

    mtx_send.lock();

    for (int s = 0; s < samples.size(); ++s) {
      ++in;
      send(sock, &samples[s].id, sizeof(uintptr_t), 0);
      send(sock, data_source->getSamplePtr(samples[s].index, 0),
           sizeof(uint64_t) * 384, 0);
      if (ncc.getVerbosityLevel())
        std::cout << ">";
    }

    mtx_send.unlock();
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

    mtx_send.lock();

    // request unique id.
    uintptr_t message = -1;
    send(sock, &message, sizeof(uintptr_t), 0);

    char buffer[128];

    assert(read_socket(sock, &buffer, sizeof(buffer)));

    uid = std::string(buffer);

    std::cout << "UID: " << uid << std::endl;

    mtx_send.unlock();

    return uid;
  }

  void ColdRun() {}

  void Receiver() {

    static float response_data[384 * 2];

    while (!terminate) {
      uintptr_t buffer;
      assert(read_socket(sock, &buffer, sizeof(uintptr_t)));
      assert(read_socket(sock, response_data, sizeof(float) * 384 * 2));
      ++out;
      // std::cout << "Received " << buffer << " " << in << " " << out <<
      // std::endl;

      std::vector<mlperf::QuerySampleResponse> responses;

      responses.push_back(
          { buffer, uintptr_t(response_data), sizeof(float) * 384 * 2 });

      mlperf::QuerySamplesComplete(responses.data(), responses.size());
      if (ncc.getVerbosityLevel())
        std::cout << "<";
    }
  }

private:
  IConfig *config;
  IDataSource *data_source;

  int sock;
  int client_fd;

  std::mutex mtx_send;

  bool terminate;
  std::thread receiver;

  std::string uid;

  NetworkClientConfig ncc;

  std::string unique_id = "TODO GET THIS FROM THE SERVER";
};

typedef KILTClient KILT;

#endif // BENCHMARK_IMPL_H
