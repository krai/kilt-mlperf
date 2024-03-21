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

#ifndef IDEVICE_H
#define IDEVICE_H

#include <iostream>

#include "idatasource.h"
#include "imodel.h"

#if defined(__amd64__) && defined(ENABLE_ZEN2)
#include <cstdint>
#include <immintrin.h>
#endif

using namespace KRAI;

template <typename Sample> class IDevice {

public:
  virtual int Inference(std::vector<Sample> samples) = 0;

  enum class State {
    READY,
    WAITING,
    ERROR
  };

  virtual State GetState() { return State::READY; }

  // Default implementation of SyncData - optimised copy from src to dest
  // Override if backend specific copy is required.
  virtual void SyncData(void * src, void * dest, int offset, size_t size){

#if defined(__amd64__) && defined(ENABLE_ZEN2)

  __m256i *dest_ptr = reinterpret_cast<__m256i *>((uint8_t *)dest+offset);

  // ensure both input and output buffers are 256 byte aligned.
  if( !(reinterpret_cast<uintptr_t>(src) & 0xff) && !(reinterpret_cast<uintptr_t>(dest_ptr) & 0xff)) {
      const __m256i *src_ptr = reinterpret_cast<const __m256i *>(src);
      int64_t vectors = size / sizeof(*src_ptr);
      for (; vectors > 0; vectors--, src_ptr++, dest_ptr++) {
        const __m256i loaded = _mm256_stream_load_si256(src_ptr);
        _mm256_stream_si256(dest_ptr, loaded);
      }
      unsigned rem = size % sizeof(*src_ptr);
      if (rem > 0) {
        memcpy((uint8_t *)dest_ptr, (uint8_t *)src_ptr, rem);
      }
      _mm_sfence();
  } else {
      // fallback for non aligned buffers
      std::copy((uint8_t *)src, (uint8_t *)src + size, (uint8_t *)dest + offset);
  }
#else
      std::copy((uint8_t *)src, (uint8_t *)src + size, (uint8_t *)dest + offset);
#endif

  };

  virtual ~IDevice(){};
};

template <typename Sample>
IDevice<Sample> *createDevice(IModel *_model, IDataSource *_data_source,
                              IConfig *_config, int hw_id,
                              std::vector<int> aff);

#endif // IDEVICE_H
