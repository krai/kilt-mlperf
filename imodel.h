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

#ifndef IMODEL_H
#define IMODEL_H

#include "idatasource.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

class IModel {
public:
  virtual void
  preprocessSamples(IDataSource *data_source, const void *samples, void *handle,
                    void (*callback)(void *handle, const void *samples)) {

    callback(handle, samples);
  }

  virtual void configureWorkload(IDataSource *data_source, const void *samples,
                                 std::vector<void *> &in_ptrs) {
    throw std::runtime_error("This variant of configWorkload() not implemented.");
  };

  virtual void configureWorkload(IDataSource *data_source, void *device, const void *samples,
                                 std::vector<void *> &in_ptrs) {
    throw std::runtime_error("This variant of configWorkload() not implemented.");
  };

  virtual void postprocessResults(void *samples,
                                  std::vector<void *> &out_ptrs) = 0;

  virtual void pipeline(void *device_ptr, IDataSource *data_source, void *samples,
                std::vector<void *> &buffers, void *metadata = nullptr) {};

  virtual int getCompletedSampleCount() const {
    return 0;
  };

  virtual ~IModel(){};
};

IModel *modelConstruct(IConfig *config);

} // namespace KRAI

#endif // IMODEL_H
