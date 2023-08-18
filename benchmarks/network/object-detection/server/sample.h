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

#include <mutex>

#define PAYLOAD_SIZE 800 * 800 * 3

struct Sample;

typedef void (*t_callback)(Sample *, uint32_t, float *);

struct Sample {

  void AllocateBuffers() {
    mem_mtx.lock();
    if (buffers.size() == 0)
      buf = aligned_alloc(256, PAYLOAD_SIZE);
    else {
      buf = buffers.back();
      buffers.pop_back();
    }
    mem_mtx.unlock();
  }

  void FreeBuffers() {
    mem_mtx.lock();
    buffers.push_back(buf);
    mem_mtx.unlock();
  }

  static std::vector<void *> buffers;
  static std::mutex mem_mtx;

  uintptr_t id;
  size_t index;
  void *buf;
  std::mutex *send_mtx;
  int sock;
  t_callback callback;
};

std::vector<void *> Sample::buffers = std::vector<void *>();
std::mutex Sample::mem_mtx;

#endif // SAMPLE_H
