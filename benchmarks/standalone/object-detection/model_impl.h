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

#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "kilt_impl.h"

#include "config/benchmark_config.h"

#include "plugins/nms-abp/nms_abp.h"

namespace KRAI {

class ResultData {
public:
  ResultData(size_t buf_size) : _size(0) { _buffer = new float[buf_size]; }
  ~ResultData() { delete[] _buffer; }
  int size() const { return _size; }
  void set_size(int size) { _size = size; }
  float *data() const { return _buffer; }

private:
  float *_buffer;
  int _size;
};

class WorkingBuffers {

public:
  WorkingBuffers(size_t buf_size, size_t b_size) {
    batch_size = b_size;
    for (int i = 0; i < batch_size; ++i) {
      nms_results.push_back(std::vector<std::vector<float>>(
          0, std::vector<float>(NUM_COORDINATES + 2, 0)));
      reformatted_results.push_back(new ResultData(buf_size));
    }
  }

  ~WorkingBuffers() {
    for (int i = 0; i < batch_size; ++i)
      delete reformatted_results[i];
  }

  void reset() {
    for (int i = 0; i < batch_size; ++i)
      nms_results[i].clear();
  }

  std::vector<std::vector<std::vector<float>>> nms_results;
  std::vector<ResultData *> reformatted_results;
  size_t batch_size;
};

template <typename TInputDataType, typename TOutput1DataType,
          typename TOutput2DataType>
class ObjectDetectionModel : public IModel {

public:
  typedef void (ObjectDetectionModel<TInputDataType, TOutput1DataType,
                                     TOutput1DataType>::*postprocessResultsPtr)(
      void *samples, std::vector<void *> &out_ptrs);

  ObjectDetectionModel(const IConfig *config) : _config(config) {

    datasource_cfg =
        static_cast<ObjectDetectionDataSourceConfig *>(_config->datasource_cfg);
    model_cfg = static_cast<ModelConfig *>(_config->model_cfg);

    nms_abp_processor =
        new NMS_ABP<TOutput1DataType, TOutput2DataType, Model_Params>(
            model_cfg->getPriorsBinPath());

    input_buf_size = datasource_cfg->getImageSize() *
                     datasource_cfg->getImageSize() *
                     datasource_cfg->getNumChannels() * sizeof(TInputDataType);

    if (model_cfg->getDeviceName() == "tensorrt")
      ppr_ptr = &KRAI::ObjectDetectionModel<
          TInputDataType, TOutput1DataType,
          TOutput1DataType>::postprocessResultsImplNMS;
    else
#ifdef MODEL_RX50
      ppr_ptr = &KRAI::ObjectDetectionModel<
          TInputDataType, TOutput1DataType,
          TOutput1DataType>::postprocessResultsImplTopK;
#else
      ppr_ptr = &KRAI::ObjectDetectionModel<
          TInputDataType, TOutput1DataType,
          TOutput1DataType>::postprocessResultsImplNoNMS;
#endif
  }

  // -------------- IModel interface BEGIN --------------------- //

  virtual void configureWorkload(IDataSource *data_source, void *device,
                                 const void *samples,
                                 std::vector<void *> &in_ptrs) {
    const std::vector<Sample> *s =
        reinterpret_cast<const std::vector<Sample> *>(samples);

    IDevice<Sample> *dev = reinterpret_cast<IDevice<Sample> *>(device);

    for (int i = 0; i < s->size(); ++i) {
      TInputDataType *src_ptr = reinterpret_cast<TInputDataType *>(
          getSamplePtr(data_source, &(*s)[i], 0));
      dev->SyncData(src_ptr, in_ptrs[0], i * input_buf_size, input_buf_size);
    }
  }

  virtual void postprocessResults(void *samples,
                                  std::vector<void *> &out_ptrs) {
    (this->*ppr_ptr)(samples, out_ptrs);
  }

  // -------------- IModel interface END ----------------------- //

private:
  virtual void *getSamplePtr(IDataSource *data_source, const Sample *s,
                             int buffer_idx) {
    return data_source->getSamplePtr(s->index, buffer_idx);
  }

  virtual void pushResult(Sample *sample, size_t size, float *result) {
#ifdef STANDALONE
    mlperf::QuerySampleResponse response(
        {sample->id, uintptr_t(result), sizeof(float) * size});
    mlperf::QuerySamplesComplete(&response, 1);
#endif
  }

  void postprocessResultsImplNMS(void *samples, std::vector<void *> &out_ptrs) {

    // const int out_vec_size = 7;

    std::vector<Sample> *s = reinterpret_cast<std::vector<Sample> *>(samples);

    if (model_cfg->getDeviceName() == "tensorrt") {
      float *outputs = reinterpret_cast<float *>(out_ptrs[0]);
      for (int i = 0; i < s->size(); i++) {
        uint32_t lastIndex = model_cfg->getOutputSize(0) - 1;
        int elementsNumber = static_cast<int>(outputs[lastIndex]);
        for (int n = 0; n < elementsNumber; n++) {
          outputs[n * out_vec_size] = (float)((*s)[i].index);
        }
        pushResult(&(*s)[i], elementsNumber * out_vec_size, outputs);
      }
    }
  }

  // No NMS - ResNet34 & SSDMobilnet
  void postprocessResultsImplNoNMS(void *samples,
                                   std::vector<void *> &out_ptrs) {

    // const int out_vec_size = 7;

    std::vector<Sample> *s = reinterpret_cast<std::vector<Sample> *>(samples);

    TOutput1DataType *boxes_ptr =
        (TOutput1DataType *)out_ptrs[modelParams.BOXES_INDEX];
    TOutput2DataType *classes_ptr =
        (TOutput2DataType *)out_ptrs[modelParams.CLASSES_INDEX];

    WorkingBuffers *wbs = popWorkingBuffers();

    for (int i = 0; i < s->size(); i++) {

      std::vector<std::vector<float>> &nms_res = wbs->nms_results[i];
      ResultData *next_result_ptr = wbs->reformatted_results[i];

      if (model_cfg->disableNMS()) {
        next_result_ptr->set_size(out_vec_size);
        next_result_ptr->data()[0] = (float)((*s)[i].index);
      } else {
        TOutput1DataType *dataLoc =
            boxes_ptr + i * modelParams.TOTAL_NUM_BOXES * NUM_COORDINATES;
        TOutput2DataType *dataConf =
            classes_ptr +
            i * modelParams.TOTAL_NUM_BOXES * modelParams.NUM_CLASSES;
        nms_abp_processor->anchorBoxProcessing(
            dataLoc, dataConf, std::ref(nms_res), (float)((*s)[i].index));

        int num_elems = nms_res.size() < (model_cfg->getMaxDetections() + 1)
                            ? nms_res.size()
                            : (model_cfg->getMaxDetections() + 1);

        next_result_ptr->set_size(num_elems * out_vec_size);
        float *buffer = next_result_ptr->data();

        for (int j = 0; j < num_elems; j++) {
          for (int i = 0; i < out_vec_size; ++i)
            *buffer++ = nms_res[j][i];
        }
      }
      pushResult(&(*s)[i], next_result_ptr->size(), next_result_ptr->data());
    }
    pushWorkingBuffers(wbs);
  }

  // TopK (Retinanet)
  void postprocessResultsImplTopK(void *samples,
                                  std::vector<void *> &out_ptrs) {

    // const int out_vec_size = 7;

    std::vector<Sample> *s = reinterpret_cast<std::vector<Sample> *>(samples);

    std::vector<TOutput1DataType *> boxes_ptrs(modelParams.OUTPUT_LEVELS);
    std::vector<TOutput2DataType *> classes_ptrs(modelParams.OUTPUT_LEVELS);
    std::vector<uint64_t *> topk_ptrs(modelParams.OUTPUT_LEVELS);

    WorkingBuffers *wbs = popWorkingBuffers();

    for (int i = 0; i < s->size(); i++) {

      std::vector<std::vector<float>> &nms_res = wbs->nms_results[i];
      ResultData *next_result_ptr = wbs->reformatted_results[i];

      if (model_cfg->disableNMS()) {
        next_result_ptr->set_size(out_vec_size);
        next_result_ptr->data()[0] = (float)((*s)[i].index);
      } else {
        for (int g = 0; g < modelParams.OUTPUT_LEVELS; ++g) {
          TOutput1DataType *boxes_ptr =
              (TOutput1DataType *)out_ptrs[modelParams.BOXES_INDEX + g];
          TOutput2DataType *classes_ptr =
              (TOutput2DataType *)out_ptrs[modelParams.CLASSES_INDEX + g];
          uint64_t *topk_ptr = (uint64_t *)out_ptrs[modelParams.TOPK_INDEX + g];

          boxes_ptrs[g] =
              boxes_ptr + i * modelParams.TOTAL_NUM_BOXES * NUM_COORDINATES;
          classes_ptrs[g] = classes_ptr + i * modelParams.TOTAL_NUM_BOXES;
          topk_ptrs[g] = topk_ptr + i * modelParams.TOTAL_NUM_BOXES;
        }
        nms_abp_processor->anchorBoxProcessing(
            (const TOutput1DataType **)boxes_ptrs.data(),
            (const TOutput2DataType **)classes_ptrs.data(),
            (const uint64_t **)topk_ptrs.data(), std::ref(nms_res),
            (float)((*s)[i].index));

        int num_elems = nms_res.size() < (model_cfg->getMaxDetections() + 1)
                            ? nms_res.size()
                            : (model_cfg->getMaxDetections() + 1);

        next_result_ptr->set_size(num_elems * out_vec_size);
        float *buffer = next_result_ptr->data();

        for (int j = 0; j < num_elems; j++) {
          for (int i = 0; i < out_vec_size; ++i)
            *buffer++ = nms_res[j][i];
        }
      }
      pushResult(&(*s)[i], next_result_ptr->size(), next_result_ptr->data());
    }
    pushWorkingBuffers(wbs);
  }

private:
  const IConfig *_config;

  ObjectDetectionDataSourceConfig *datasource_cfg;
  ModelConfig *model_cfg;

  const int out_vec_size = 7;

  int _current_buffer_size = 0;

  NMS_ABP<TOutput1DataType, TOutput2DataType, Model_Params> *nms_abp_processor;
  Model_Params modelParams;

  postprocessResultsPtr ppr_ptr;

  WorkingBuffers *popWorkingBuffers() {
    working_buffs_mtx.lock();
    WorkingBuffers *tmp;
    if (working_buffers_list.size() == 0) {
      tmp =
          new WorkingBuffers((model_cfg->getMaxDetections() + 1) * out_vec_size,
                             model_cfg->getBatchSize());
    } else {
      tmp = working_buffers_list.back();
      working_buffers_list.pop_back();
    }
    working_buffs_mtx.unlock();
    return tmp;
  }

  void pushWorkingBuffers(WorkingBuffers *bufs) {
    working_buffs_mtx.lock();
    bufs->reset();
    working_buffers_list.push_back(bufs);
    working_buffs_mtx.unlock();
  }

  uint32_t input_buf_size;

  std::vector<WorkingBuffers *> working_buffers_list;
  std::mutex working_buffs_mtx;
};

} // namespace KRAI

#endif // MODEL_IMPL_H
