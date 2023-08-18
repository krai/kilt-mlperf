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

#define SEPARATOR 102

bool trace = false;

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

class Server {
public:
  Server() {

    trace = (nsc.getVerbosityLevel() > 0);

    std::cout << "Constructing Server..." << std::endl;
    int opt = 1;

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      perror("socket failed");
      exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
      perror("setsockopt");
      exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(nsc.getNetworkServerPort());

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
      perror("bind failed");
      exit(EXIT_FAILURE);
    }
    std::cout << "Server Constructed." << std::endl;
  }

  ~Server() {
    // closing the connected socket
    close(sock);
    // closing the listening socket
    shutdown(server_fd, SHUT_RDWR);
  }

  void Run() {

    SizedSample dummy;
    std::vector<SizedSample> samples;
    samples.push_back(dummy);
    Sample *sample = &(samples[0].first);

    enum State { CONNECT, SERVE };

    static char uid_string[128];
    strcpy(uid_string, kil.UniqueServerID().c_str());

    State state = CONNECT;

    while (true) {

      switch (state) {

      case CONNECT: {

        std::cout << "Waiting for connection..." << std::endl;
        if (listen(server_fd, 3) < 0) {
          perror("listen");
          exit(EXIT_FAILURE);
        }

        int addrlen = sizeof(address);

        if ((sock = accept(server_fd, (struct sockaddr *)&address,
                           (socklen_t *)&addrlen)) < 0) {
          perror("accept");
          exit(EXIT_FAILURE);
        }
        state = SERVE;
        std::cout << "Connected." << std::endl;
      };

      case SERVE: {

        while (true) {

          uintptr_t message_header;
          assert(read_socket(sock, &message_header, sizeof(uintptr_t)));

          if (message_header == -1) {
            std::cout << "Sending UID..." << std::endl;
            // reply with name
            send_mtx.lock();
            send(sock, uid_string, sizeof(uid_string), 0);
            send_mtx.unlock();
          } else if (message_header == -2) {
            // disconnect
            std::cout << std::endl << "Disconnecting." << std::endl;
            close(sock);
            state = CONNECT;
            break;
          } else { // must be a sample

            sample->AllocateBuffers();

            sample->id = message_header;
            sample->sock = sock;
            sample->send_mtx = &send_mtx;
            sample->callback = Server::Callback;

            assert(read_socket(sock, sample->buf0, 384 * sizeof(uint64_t)));

            // Generate the second and third buffers from the first.
            memset(sample->buf1, 0, 384 * sizeof(uint64_t));
            memset(sample->buf2, 0, 384 * sizeof(uint64_t));

            int idx = 0;
            for (int x = 0; x < 2; ++x) {
              do {
                sample->buf1[idx] = 1;
                sample->buf2[idx] = x;
              } while (sample->buf0[idx++] != SEPARATOR);
            }

            if (trace)
              std::cout << ">";

            kil.Inference(samples);
          }
        }
      };
      }
    }
  }

  static void Callback(Sample *sample, float *data) {

    sample->send_mtx->lock();
    send(sample->sock, &(sample->id), sizeof(uintptr_t), 0);
    send(sample->sock, data, 384 * sizeof(float) * 2, 0);
    sample->send_mtx->unlock();
    if (trace)
      std::cout << "<";

    sample->FreeBuffers();
  }

private:
  KraiInferenceLibrary<SizedSample> kil;
  int server_fd, sock;
  std::mutex send_mtx;
  struct sockaddr_in address;
  NetworkServerConfig nsc;
};

int main(int argc, char const *argv[]) {
  std::cout << "Creating Server" << std::endl;
  Server server;
  std::cout << "Calling Run" << std::endl;
  server.Run();
  std::cout << "End" << std::endl;
}
