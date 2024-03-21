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

#pragma once

#include <netinet/in.h>
#include <vector>
#include <utility>

void init_server(int port);
void init_client(const char *ip_addr, int port);

class Connection {
public:
  virtual ~Connection() = 0;

  bool read(void *dest, size_t len);
  bool read(void *dest, size_t len, const bool &cancel);

  bool write(const void *src, size_t len);

  // Optimised version of write that prevents fragmentation
  bool write(std::vector<std::pair<const void *, size_t>> data);

protected:
  int sock;
  int fd;
};

class ServerConnection : public Connection {
public:
  ServerConnection();
  ~ServerConnection();
};

class ClientConnection : public Connection {
public:
  ClientConnection();
  ~ClientConnection();
};
