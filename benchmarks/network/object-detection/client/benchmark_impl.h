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

#include "config/benchmark_config.h"
#include "datasource.h"

#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <queue>

#define PAYLOAD_SIZE 800 * 800 * 3

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

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(ncc.getNetworkServerPort());

    if (inet_pton(AF_INET, ncc.getNetworkServerIPAddress().c_str(),
                  &serv_addr.sin_addr) <= 0) {
      std::cout << "Address invalid / not supported." << std::endl;
      exit(1);
    }

    client_fds.resize(ncc.getNumSockets());
    socks.resize(ncc.getNumSockets());

    for (int s = 0; s < ncc.getNumSockets(); ++s) {

      if ((socks[s] = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cout << "Could not create socket." << std::endl;
        exit(1);
      }

      while ((client_fds[s] = connect(socks[s], (struct sockaddr *)&serv_addr,
                                      sizeof(serv_addr))) < 0) {
        std::cout << "Waiting to connect..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
      std::cout << "Connected (" << s << ")" << std::endl;
    }

    receiver_sock = GetSock();
    std::cout << "Reciever Sock: " << receiver_sock << std::endl;

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

    int sock = GetSock();
    // tell server to detach
    uintptr_t message = -2;
    send(sock, &message, sizeof(uintptr_t), 0);
    ReleaseSock(sock);

    // closing the connected socket
    for (int s = 0; s < ncc.getNumSockets(); ++s) {
      close(client_fds[s]);
    }
  }

  void Inference(const std::vector<mlperf::QuerySample> &samples) {

    for (int s = 0; s < samples.size(); ++s) {
      samples_queue.push(samples[s]);
    }
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

      int sock = GetSock();
      // std::cout << "Got Sock." << std::endl;
      send(sock, &sample.id, sizeof(uintptr_t), 0);
      // std::cout << "Sending ID " << sock << std::endl;
      send(sock, &sample.index, sizeof(size_t), 0);
      // std::cout << "Sending Index " << sock << std::endl;
      send(sock, data_source->getSamplePtr(sample.index, 0), PAYLOAD_SIZE, 0);
      // std::cout << "Sending Data " << sock << std::endl;
      if (ncc.getVerbosityLevel())
        std::cout << ">";

      ReleaseSock(sock);
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

    std::cout << "Requesting Unique Server ID" << std::endl;

    int sock = GetSock();

    // request unique id.
    uintptr_t message = -1;
    send(sock, &message, sizeof(uintptr_t), 0);

    char buffer[128];

    assert(read_socket(sock, &buffer, sizeof(buffer)));

    uid = std::string(buffer);

    std::cout << "UID: " << uid << " - sock: " << sock << std::endl;

    ReleaseSock(sock);

    return uid;
  }

  void ColdRun() {}

  void Receiver() {

    static float response_data[65536]; // unfeasibly big number

    while (!terminate) {
      uintptr_t id;
      uint32_t length;
      // std::cout << "reviever " << receiver_sock << std::endl;
      assert(read_socket(receiver_sock, &id, sizeof(uintptr_t)));
      assert(read_socket(receiver_sock, &length, sizeof(uint32_t)));
      assert(read_socket(receiver_sock, response_data, sizeof(float) * length));
      ++out;
      // std::cout << "Received " << buffer << " " << in << " " << out <<
      // std::endl;

      std::vector<mlperf::QuerySampleResponse> responses;

      responses.push_back(
          {id, uintptr_t(response_data), sizeof(float) * length});

      mlperf::QuerySamplesComplete(responses.data(), responses.size());
      if (ncc.getVerbosityLevel())
        std::cout << "<";
    }

    ReleaseSock(receiver_sock);
  }

private:
  int GetSock() {

    int sock = -1;
    while (sock == -1) {
      mtx_send.lock();
      if (socks.size() != 0) {
        sock = socks.back();
        // std::cout << "GetSock: " << socks.size() << " " << sock << std::endl;
        socks.pop_back();
      }
      mtx_send.unlock();
    }
    return sock;
  }

  void ReleaseSock(int sock) {
    mtx_send.lock();
    socks.push_back(sock);
    // std::cout << "ReleaseSock: " << socks.size() << " " << sock << std::endl;
    mtx_send.unlock();
  }

  IConfig *config;
  IDataSource *data_source;

  std::vector<int> socks;
  std::vector<int> client_fds;
  int receiver_sock;

  std::mutex mtx_send;

  std::queue<mlperf::QuerySample> samples_queue;
  std::mutex samples_mtx;

  bool terminate;
  std::thread receiver;

  std::vector<std::thread> transmitters;

  std::string uid;

  NetworkClientConfig ncc;
};

typedef KILTClient KILT;

#endif // BENCHMARK_IMPL_H
