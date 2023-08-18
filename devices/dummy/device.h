//
// MIT License
//
// Copyright (c) 2023 Krai Ltd
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

#ifndef DEVICE_H
#define DEVICE_H

#include "config/device_config.h"
#include "idatasource.h"
#include "imodel.h"

#define LARGE_BUFFER 4000000

using namespace KRAI;

template <typename Sample> class Device : public IDevice<Sample> {

public:
  void Construct(IModel *_model, IDataSource *_data_source, IConfig *_config,
                 int hw_id, std::vector<int> aff) {

    model = _model;
    data_source = _data_source;

    model_cfg = static_cast<IModelConfig *>(_config->model_cfg);

    // load model from config path
    // TODO

    // create dummy input buffers for device
    for (int i = 0; i < model_cfg->getInputCount(); ++i) {
      char *buf = new char[LARGE_BUFFER];
      buf =
          (char *)((((uint64_t)buf) / 256) * 256); // align to 256 byte boundary
      buffers_in.push_back(buf);
    }

    // create dummy output buffers for device
    for (int i = 0; i < model_cfg->getOutputCount(); ++i)
      buffers_out.push_back(new char[LARGE_BUFFER]);
  }

  virtual int Inference(std::vector<Sample> samples) {

    // populate device input buffers from datasource
    model->configureWorkload(data_source, &samples, buffers_in);

    // do device specific inference here
    // TODO

    // pass device output buffers to model specific post processing
    model->postprocessResults(&samples, buffers_out);

    return 0;
  }

  ~Device() {}

private:
  IModelConfig *model_cfg;

  // activations, set, input buffers
  std::vector<void *> buffers_in;

  // activation, set, output buffers
  std::vector<void *> buffers_out;

  IModel *model;
  IDataSource *data_source;
};

template <typename Sample>
IDevice<Sample> *createDevice(IModel *_model, IDataSource *_data_source,
                              IConfig *_config, int hw_id,
                              std::vector<int> aff) {
  Device<Sample> *d = new Device<Sample>();
  d->Construct(_model, _data_source, _config, hw_id, aff);
  return d;
}

#endif // DEVICE_H
