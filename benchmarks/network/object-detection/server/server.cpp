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

#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sample.h"

bool trace = false;

class Server {
public:
  Server() {

    trace = (nsc.getVerbosityLevel() > 0);

    std::cout << "Constructing Server..." << std::endl;
    int opt = 1;

    server_fds.resize(nsc.getNumSockets());
    socks.resize(nsc.getNumSockets());

    // Creating socket file descriptor
    if ((server_fds[0] = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      perror("socket failed");
      exit(EXIT_FAILURE);
    }
    std::cout << "server_fds " << server_fds[0] << std::endl;

    if (setsockopt(server_fds[0], SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
      perror("setsockopt");
      exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(nsc.getNetworkServerPort());

    if (bind(server_fds[0], (struct sockaddr *)&address, sizeof(address)) < 0) {
      perror("bind failed");
      exit(EXIT_FAILURE);
    }

    state = CONNECT;

    for (int t = 0; t < nsc.getNumSockets(); ++t) {
      serve_threads.push_back(std::thread(&Server::ServeThread, this));
    }

    std::cout << "Server Constructed." << std::endl;
  }

  ~Server() {
    for (int s = 0; s < nsc.getNumSockets(); ++s) {
      // closing the connected socket
      close(socks[s]);
      // closing the listening socket
      // shutdown(server_fds[s], SHUT_RDWR);
    }
  }

  void Run() {

    while (true) {

      switch (state) {

      case DISCONNECT: {
        ReleaseSock(transmitter_sock);
        CloseSocks();
        state = CONNECT;
      };

      case CONNECT: {

        for (int s = 0; s < nsc.getNumSockets(); ++s) {
          std::cout << "Waiting for connection... " << s << std::endl;
          if (listen(server_fds[0], 8) < 0) {
            perror("listen");
            exit(EXIT_FAILURE);
          }

          int addrlen = sizeof(address);

          if ((socks[s] = accept(server_fds[0], (struct sockaddr *)&address,
                                 (socklen_t *)&addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
          }
        }

        transmitter_sock = GetSock();
        std::cout << "Transmitter sock: " << transmitter_sock << std::endl;

        state = SERVE;
        std::cout << "Connected." << std::endl;
      };

      case SERVE: {
        // ServeThread();
        //  sleep as covered by threads
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      }
    }
  }

  void ServeThread() {

    std::cout << "Creating ServeThread" << std::endl;

    Sample dummy;
    std::vector<Sample> samples;
    samples.push_back(dummy);
    Sample *sample = &(samples[0]);

    static char uid_string[128];
    strcpy(uid_string, kil.UniqueServerID().c_str());

    while (true) {

      std::this_thread::yield();

      if (state == SERVE) {

        int sock = GetSock();

        uintptr_t message_header;
        // std::cout << "Reading message header " << sock << std::endl;
        if (ReadSock(sock, &message_header, sizeof(uintptr_t))) {

          if (message_header == -1) {
            std::cout << "Sending UID... " << sock << std::endl;
            // reply with name
            send(sock, uid_string, sizeof(uid_string), 0);
          } else if (message_header == -2) {
            // disconnect
            std::cout << std::endl << "Disconnecting." << std::endl;
            state = DISCONNECT;
          } else { // must be a sample

            sample->AllocateBuffers();

            sample->id = message_header;
            sample->sock = transmitter_sock;
            sample->send_mtx = &mtx_tx;
            sample->callback = Server::Callback;

            // std::cout << "Reading Sample index. " << sock << std::endl;
            assert(ReadSock(sock, &sample->index, sizeof(size_t)));

            // std::cout << "Reading Sample data." << sock << std::endl;
            assert(ReadSock(sock, sample->buf, PAYLOAD_SIZE));

            if (trace)
              std::cout << ">";

            // std::cout << "Calling inference " << sock << std::endl;
            kil.Inference(samples);
          }
        }
        ReleaseSock(sock);
      }
    }
  }

  static void Callback(Sample *sample, uint32_t length, float *data) {

    // std::cout << "Sending result " << sample->sock << std::endl;
    sample->send_mtx->lock();
    send(sample->sock, &(sample->id), sizeof(uintptr_t), 0);
    send(sample->sock, &length, sizeof(uint32_t), 0);
    send(sample->sock, data, length * sizeof(float), 0);
    sample->send_mtx->unlock();
    if (trace)
      std::cout << "<";
    // std::cout << "Result sent " << sample->sock << std::endl;

    sample->FreeBuffers();
  }

private:
  enum State { DISCONNECT, CONNECT, SERVE };

  int GetSock() {

    int sock = -1;
    while (sock == -1) {
      mtx_sock.lock();
      if (socks.size() != 0) {
        sock = socks.back();
        // std::cout << "GetSock: " << socks.size() << " " << sock << std::endl;
        socks.pop_back();
      }
      mtx_sock.unlock();
    }
    return sock;
  }

  void ReleaseSock(int sock) {
    mtx_sock.lock();
    socks.push_back(sock);
    // std::cout << "ReleaseSock: " << socks.size() << " " << sock << std::endl;
    mtx_sock.unlock();
  }

  void CloseSocks() {
    bool complete = false;
    while (!complete) {
      mtx_sock.lock();
      // if all the sockets have now unused
      if (socks.size() == nsc.getNumSockets()) {
        // close them
        for (int s = 0; s < nsc.getNumSockets(); ++s) {
          close(socks[s]);
          complete = true;
        }
      }
      mtx_sock.unlock();
    }
  }

  bool ReadSock(int sock, void *buf, size_t len) {

    uint8_t *byte_buf = reinterpret_cast<uint8_t *>(buf);

    while (len != 0) {

      if (state == DISCONNECT)
        return false;

      size_t bytes = read(sock, byte_buf, len);
      assert(bytes >= 0);

      byte_buf += bytes;

      len -= bytes;
    }
    return true;
  }

  KraiInferenceLibrary<Sample> kil;
  std::vector<int> socks;
  std::vector<int> server_fds;
  int transmitter_sock;

  std::mutex mtx_sock;
  std::mutex mtx_tx;
  struct sockaddr_in address;
  NetworkServerConfig nsc;
  std::vector<std::thread> serve_threads;
  State state;
};

int main(int argc, char const *argv[]) {
  std::cout << "Creating Server" << std::endl;
  Server server;
  std::cout << "Calling Run" << std::endl;
  server.Run();
  std::cout << "End" << std::endl;
}
