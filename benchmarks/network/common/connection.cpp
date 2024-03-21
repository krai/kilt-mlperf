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

#include "connection.h"

#include <iostream>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>

static struct sockaddr_in address;
static int server_fd; // Only exists for server connections

void init_server(int port) {
  // Creating socket file descriptor
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  int opt = 1;
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt))) {
    perror("setsockopt");
    exit(EXIT_FAILURE);
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }
}

void init_client(const char *ip_addr, int port) {
  address.sin_family = AF_INET;
  address.sin_port = htons(port);

  if (inet_pton(AF_INET, ip_addr, &address.sin_addr) <= 0) {
    std::cout << "Address invalid / not supported." << std::endl;
    exit(1);
  }
}

bool Connection::read(void *dest, size_t len) {
  bool cancel = false;
  return read(dest, len, cancel);
}

bool Connection::read(void *dest, size_t len, const bool &cancel) {
  uint8_t *byte_buf = reinterpret_cast<uint8_t *>(dest);

  while (len != 0) {
    if (cancel) {
      return false;
    }

    size_t bytes = ::read(sock, byte_buf, len);
    if (bytes < 0)
      return false;

    byte_buf += bytes;

    len -= bytes;
  }
  return true;
}

bool Connection::write(const void *src, size_t len) {
  return ::send(sock, src, len, 0) != -1;
}

bool Connection::write(std::vector<std::pair<const void *, size_t>> data) {
  for (int i = 0; i < data.size() - 1; i++) {
    auto [src, len] = data[i];
    if (!::send(sock, src, len, MSG_MORE))
      return false;
  }
  auto [src, len] = data[data.size() - 1];
  return ::send(sock, src, len, 0);
}

ServerConnection::ServerConnection() {
  if (listen(server_fd, 8) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }

  int addrlen = sizeof(address);

  if ((sock = ::accept(server_fd, (struct sockaddr *)&address,
                       (socklen_t *)&addrlen)) < 0) {
    perror("accept");
    exit(EXIT_FAILURE);
  }
}

ServerConnection::~ServerConnection() { close(sock); }

ClientConnection::ClientConnection() {
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    perror("socket creation failed");
    exit(1);
  }

  while ((fd = connect(sock, (struct sockaddr *)&address, sizeof(address))) <
         0) {
    std::cout << "Waiting to connect..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
}

ClientConnection::~ClientConnection() { close(fd); }

Connection::~Connection() {}