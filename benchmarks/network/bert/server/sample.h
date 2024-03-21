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

#ifndef SAMPLE_H
#define SAMPLE_H

#include "benchmarks/network/common/connection.h"
#include "benchmarks/network/common/connection.h"
#include <mutex>
#include <cassert>
#include <string.h>
#include <cassert>
#include <string.h>

#define SEPARATOR 102

struct Sample {

  void AllocateBuffers() {
    buf0 = new uint64_t[384];
    buf1 = new uint64_t[384];
    buf2 = new uint64_t[384];
  }

  void FreeBuffers() {
    delete buf0;
    delete buf1;
    delete buf2;
  }

  template <typename TInputDataType>
  void ReadSample(ServerConnection &conn, bool &cancel) {
    assert(conn.read(&index, sizeof(size_t)));

    uint64_t tmp_buf[384];

    assert(conn.read(tmp_buf, 384 * sizeof(uint64_t), cancel));
    for (int x = 0; x < 384; ++x)
      reinterpret_cast<TInputDataType *>(buf0)[x] = tmp_buf[x];

    // Generate the second and third buffers from the first.
    memset(buf1, 0, 384 * sizeof(uint64_t));
    memset(buf2, 0, 384 * sizeof(uint64_t));

    int idx = 0;
    for (int x = 0; x < 2; ++x) {
      do {
        reinterpret_cast<TInputDataType *>(buf1)[idx] = 1;
        reinterpret_cast<TInputDataType *>(buf2)[idx] = x;
      } while (reinterpret_cast<TInputDataType *>(buf0)[idx++] != SEPARATOR);
    }
  }

  void Callback(float *data) {
    const uint32_t reply_length = 382 * 2;
    send_mtx->lock();

    reply_conn->write({{&id, sizeof(uintptr_t)},
                       {&reply_length, sizeof(uint32_t)},
                       {data, reply_length * sizeof(float)}});

    send_mtx->unlock();
    FreeBuffers();
  }

  uintptr_t id;
  uintptr_t index;
  uint64_t *buf0;
  uint64_t *buf1;
  uint64_t *buf2;
  std::mutex *send_mtx;
  ServerConnection *reply_conn;
};

typedef std::pair<Sample, int> SizedSample;

#endif // SAMPLE_H
