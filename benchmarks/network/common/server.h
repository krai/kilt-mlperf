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

#include "kilt.h"
#include "iconfig.h"

#include "benchmarks/network/common/connection.h"
#include "benchmarks/network/common/config/network_config.h"

#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <mutex>
#include <memory>

bool trace = false;

template <typename TInputDataType, typename TSampleType> class Server {
public:
  Server() {
    trace = (nsc.getVerbosityLevel() > 0);

    std::cout << "Constructing Server..." << std::endl;
    state = CONNECT;

    init_server(nsc.getNetworkServerPort());

    for (int t = 0; t < nsc.getNumSockets(); ++t) {
      serve_threads.push_back(std::thread(&Server::ServeThread, this));
    }

    std::cout << "Server Constructed." << std::endl;
  }

  ~Server() {
    // closing the listening socket
    // shutdown(server_fd, SHUT_RDWR);
  }

  void Run() {
    while (true) {

      switch (state) {

      case DISCONNECT: {
        ReleaseConn(std::move(transmitter_conn));
        CloseConns();
        state = CONNECT;
        is_disconnecting = false;
      }

      case CONNECT: {
        for (int s = 0; s < nsc.getNumSockets(); ++s) {
          std::cout << "Waiting for connection..." << std::endl;
          conns.push_back(std::make_unique<ServerConnection>());
        };

        transmitter_conn = GetConn();

        state = SERVE;
        std::cout << "Connected." << std::endl;
      }

      case SERVE: {
        //  sleep as covered by threads
        std::this_thread::sleep_for(std::chrono::seconds(1));
      };
      }
    }
  }

  void ServeThread() {
    std::cout << "Creating ServeThread" << std::endl;

    TSampleType dummy;
    std::vector<TSampleType> samples;
    samples.push_back(dummy);
    Sample *sample;

    // TODO: find a good way to do this if it exists
    if constexpr (std::is_same_v<TSampleType,
                                 std::pair<Sample, int>>) { // SizedSample
      sample = &(samples[0].first);
    } else if constexpr (std::is_same_v<TSampleType, Sample>) {
      sample = &(samples[0]);
    }

    static char uid_string[128];
    strcpy(uid_string, kil.UniqueServerID().c_str());

    while (true) {
      std::this_thread::yield();

      if (state == SERVE) {
        auto conn = GetConn();

        uintptr_t message_header;
        if (conn->read(&message_header, sizeof(uintptr_t), is_disconnecting)) {
          if (message_header == -1) {
            std::cout << "Sending UID..." << std::endl;
            // reply with name
            assert(conn->write(uid_string, sizeof(uid_string)));

          } else if (message_header == -2) {
            // disconnect
            std::cout << std::endl << "Disconnecting." << std::endl;
            state = DISCONNECT;
            is_disconnecting = true;
          } else { // must be a sample

            sample->AllocateBuffers();

            sample->id = message_header;
            sample->reply_conn = transmitter_conn.get();
            sample->send_mtx = &mtx_tx;

            sample->ReadSample<TInputDataType>(*conn, is_disconnecting);

            if (trace)
              std::cout << ">";

            kil.Inference(samples);
          }
        }
        ReleaseConn(std::move(conn));
      }
    }
  }

private:
  enum State { DISCONNECT, CONNECT, SERVE };

  std::unique_ptr<ServerConnection> GetConn() {
    std::unique_ptr<ServerConnection> conn = nullptr;

    while (conn == nullptr) {
      mtx_sock.lock();
      if (conns.size() != 0) {
        conn = std::move(conns.back());
        conns.pop_back();
      }
      mtx_sock.unlock();
    }

    return conn;
  }

  void ReleaseConn(std::unique_ptr<ServerConnection> conn) {
    mtx_sock.lock();
    conns.push_back(std::move(conn));
    mtx_sock.unlock();
  }

  void CloseConns() {
    bool complete = false;
    while (!complete) {
      mtx_sock.lock();
      // if all the sockets have now unused
      if (conns.size() == nsc.getNumSockets()) {
        conns.clear();
        complete = true;
      }
      mtx_sock.unlock();
    }
  }

  KraiInferenceLibrary<TSampleType> kil;

  std::vector<std::unique_ptr<ServerConnection>> conns;

  std::unique_ptr<ServerConnection> transmitter_conn;

  std::mutex mtx_sock;
  std::mutex mtx_tx;

  NetworkServerConfig nsc;

  State state;
  std::vector<std::thread> serve_threads;

  bool is_disconnecting = false;
};
