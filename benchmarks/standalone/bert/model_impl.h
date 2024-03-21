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

#ifndef BERT_MODEL_IMPL_H
#define BERT_MODEL_IMPL_H

#include <stdio.h>
#include <stdlib.h>

#include "config/benchmark_config.h"

#include "kilt_impl.h"

#include "pack.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace KRAI {

typedef void (*callbackPtr)(void *handle, const void *samples);

template <typename TInputDataType, typename TOutputDataType>
class BertModel : public IModel {

public:
  typedef void (
      BertModel<TInputDataType, TOutputDataType>::*preprocessSamplesPtr)(
      IDataSource *data_source, const void *samples, void *handle, callbackPtr);
  typedef void (BertModel<TInputDataType, TOutputDataType>::*
                    configureWorkloadPtr)(IDataSource *data_source,
                                          void *device, const void *samples,
                                          std::vector<void *> &in_ptrs);
  typedef void (
      BertModel<TInputDataType, TOutputDataType>::*postprocessResultsPtr)(
      void *samples, std::vector<void *> &out_ptrs);

  BertModel(const IConfig *config) : _config(config) {

    BertModelConfig::BERT_MODEL_VARIANT bmv =
        static_cast<BertModelConfig *>(_config->model_cfg)->getModelVariant();

    if (bmv == BertModelConfig::BERT_ORIG) {
      pps_ptr = &KRAI::BertModel<TInputDataType,
                                 TOutputDataType>::preprocessSamplesImpl;
      cw_ptr = &KRAI::BertModel<TInputDataType,
                                TOutputDataType>::configureWorkloadOrigImpl;
      ppr_ptr = &KRAI::BertModel<TInputDataType,
                                 TOutputDataType>::postprocessResultsImpl;
    } else if (bmv == BertModelConfig::BERT_PACKED) {
      pps_ptr = &KRAI::BertModel<TInputDataType,
                                 TOutputDataType>::preprocessSamplesImpl;
      cw_ptr = &KRAI::BertModel<TInputDataType,
                                TOutputDataType>::configureWorkloadPackedImpl;
      ppr_ptr = &KRAI::BertModel<TInputDataType,
                                 TOutputDataType>::postprocessResultsImpl;
    } else if (bmv == BertModelConfig::DISTILBERT_PACKED) {
      pps_ptr = &KRAI::BertModel<TInputDataType,
                                 TOutputDataType>::preprocessSamplesImpl;
      cw_ptr = &KRAI::BertModel<TInputDataType, TOutputDataType>::
                   configureWorkloadDistilBERTPackedImpl;
      ppr_ptr = &KRAI::BertModel<TInputDataType,
                                 TOutputDataType>::postprocessResultsImpl;
    } else {
      throw("Unknown BERT model type.");
    }

    packed_seq_len = static_cast<BertModelConfig *>(_config->model_cfg)
                         ->getModelSequenceLength();

    datasource_seq_len =
        static_cast<SquadDataSourceConfig *>(_config->datasource_cfg)
            ->getDataSourceSequenceLength();

    _zero_buffer = std::vector<TInputDataType>(packed_seq_len, 0);
  }

  // -------------- IModel interface BEGIN --------------------- //
  virtual void
  preprocessSamples(IDataSource *data_source, const void *samples, void *handle,
                    void (*callback)(void *handle, const void *samples)) {
    (this->*pps_ptr)(data_source, samples, handle, callback);
  }

  virtual void configureWorkload(IDataSource *data_source, void *device,
                                 const void *samples,
                                 std::vector<void *> &in_ptrs) override {
    (this->*cw_ptr)(data_source, device, samples, in_ptrs);
  }

  virtual void postprocessResults(void *samples,
                                  std::vector<void *> &out_ptrs) {
    (this->*ppr_ptr)(samples, out_ptrs);
  }
  // -------------- IModel interface END ----------------------- //

private:
  virtual void *getSamplePtr(IDataSource *data_source, const SizedSample *s,
                             int sample_idx, int buffer_idx) {
    return data_source->getSamplePtr(sample_idx, buffer_idx);
  }

  virtual void preprocessSamplesImpl(IDataSource *data_source,
                                     const void *samples, void *handle,
                                     void (*callback)(void *handle,
                                                      const void *samples)) {

    std::vector<SizedSample> sm =
        *(reinterpret_cast<const std::vector<SizedSample> *>(samples));

    // get the sizes of each of the inputs
    for (int s = 0; s < sm.size(); ++s) {

      // get the input mask
      TInputDataType *src = static_cast<TInputDataType *>(
          getSamplePtr(data_source, &(sm[s]), sm[s].first.index, 1));

      int sum = 0;
      for (int j = 0; j < datasource_seq_len; ++j) {
        sum += src[j];
      }
      sm[s].second = sum;
    }

    std::vector<std::vector<SizedSample>> packed_samples;

    pack(sm, packed_seq_len, 3, packed_samples);

    for (int ps = 0; ps < packed_samples.size(); ++ps) {
      callback(handle, &packed_samples[ps]);
    }
  }

  void configureWorkloadOrigImpl(IDataSource *data_source, void *device,
                                 const void *samples,
                                 std::vector<void *> &in_ptrs) {

    const std::vector<SizedSample> *sm =
        reinterpret_cast<const std::vector<SizedSample> *>(samples);

    IDevice<SizedSample> *d = reinterpret_cast<IDevice<SizedSample> *>(device);

    packed_seq_len = 384;

    TInputDataType *src0 = static_cast<TInputDataType *>(
        getSamplePtr(data_source, &(*sm)[0], (*sm)[0].first.index, 0));
    TInputDataType *src1 = static_cast<TInputDataType *>(
        getSamplePtr(data_source, &(*sm)[0], (*sm)[0].first.index, 1));
    TInputDataType *src2 = static_cast<TInputDataType *>(
        getSamplePtr(data_source, &(*sm)[0], (*sm)[0].first.index, 2));

    d->SyncData(src0, in_ptrs[0], 0, packed_seq_len * sizeof(TInputDataType));
    d->SyncData(src1, in_ptrs[1], 0, packed_seq_len * sizeof(TInputDataType));
    d->SyncData(src2, in_ptrs[2], 0, packed_seq_len * sizeof(TInputDataType));
  }

  void configureWorkloadPackedImpl(IDataSource *data_source, void *device,
                                   const void *samples,
                                   std::vector<void *> &in_ptrs) {

    const std::vector<SizedSample> *sm =
        reinterpret_cast<const std::vector<SizedSample> *>(samples);

    IDevice<SizedSample> *d = reinterpret_cast<IDevice<SizedSample> *>(device);

    unsigned int offset = 0;

    for (int s = 0; s < sm->size(); ++s) {

      TInputDataType *src0 = static_cast<TInputDataType *>(
          getSamplePtr(data_source, &(*sm)[s], (*sm)[s].first.index, 0));
      TInputDataType *src2 = static_cast<TInputDataType *>(
          getSamplePtr(data_source, &(*sm)[s], (*sm)[s].first.index, 2));

      TInputDataType sample_seq_len = (*sm)[s].second;

      d->SyncData(src0, in_ptrs[0], offset * sizeof(TInputDataType),
                  sample_seq_len * sizeof(TInputDataType));
      d->SyncData(src2, in_ptrs[2], offset * sizeof(TInputDataType),
                  sample_seq_len * sizeof(TInputDataType));

      for (TInputDataType m = 0; m < sample_seq_len; m++) {
        d->SyncData(&m, in_ptrs[3], offset * sizeof(TInputDataType),
                    sizeof(TInputDataType));
        ++offset;
      }

      d->SyncData(&sample_seq_len, in_ptrs[1], s * sizeof(TInputDataType),
                  sizeof(TInputDataType));
    }

    // Zero the remainder of the buffers
    d->SyncData(_zero_buffer.data(), in_ptrs[0],
                offset * sizeof(TInputDataType),
                (packed_seq_len - offset) * sizeof(TInputDataType));
    d->SyncData(_zero_buffer.data(), in_ptrs[2],
                offset * sizeof(TInputDataType),
                (packed_seq_len - offset) * sizeof(TInputDataType));
    d->SyncData(_zero_buffer.data(), in_ptrs[3],
                offset * sizeof(TInputDataType),
                (packed_seq_len - offset) * sizeof(TInputDataType));
    d->SyncData(_zero_buffer.data(), in_ptrs[1],
                sm->size() * sizeof(TInputDataType),
                (8 - sm->size()) * sizeof(TInputDataType));
  }

  void configureWorkloadDistilBERTPackedImpl(IDataSource *data_source,
                                             void *device, const void *samples,
                                             std::vector<void *> &in_ptrs) {

    const std::vector<SizedSample> *sm =
        reinterpret_cast<const std::vector<SizedSample> *>(samples);

    // clear input buffers
    memset(in_ptrs[0], 0, packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[1], 0,
           packed_seq_len * packed_seq_len * sizeof(TInputDataType));
    memset(in_ptrs[2], 0, packed_seq_len * sizeof(TInputDataType));

    unsigned int offset = 0;

    for (int s = 0; s < sm->size(); ++s) {

      TInputDataType *src0 = static_cast<TInputDataType *>(
          getSamplePtr(data_source, &(*sm)[s], (*sm)[s].first.index, 0));

      TInputDataType sample_seq_len = (*sm)[s].second;

      applyMask(static_cast<TInputDataType *>(in_ptrs[1]), sample_seq_len,
                offset);

      for (int m = 0; m < sample_seq_len; m++) {
        static_cast<TInputDataType *>(in_ptrs[0])[offset] = src0[m];
        static_cast<TInputDataType *>(in_ptrs[2])[offset] = m;
        ++offset;
      }
    }
  }

  virtual void pushResult(SizedSample *sample, std::vector<float> &result) {
#ifdef STANDALONE
    mlperf::QuerySampleResponse response({sample->first.id,
                                          uintptr_t(&result[0]),
                                          sizeof(float) * result.size()});
    mlperf::QuerySamplesComplete(&response, 1);
#endif
  }

  virtual void postprocessResultsImpl(void *samples,
                                      std::vector<void *> &out_ptrs) {

    std::vector<SizedSample> *sm =
        reinterpret_cast<std::vector<SizedSample> *>(samples);

    int offset = 0;

    for (int i = 0; i < sm->size(); ++i) {

      TInputDataType sample_seq_len = (*sm)[i].second;

      std::vector<float> result(datasource_seq_len * 2, -10000.0f);

      TOutputDataType *b0 = ((TOutputDataType *)out_ptrs[0]) + offset;
      TOutputDataType *b1 = ((TOutputDataType *)out_ptrs[1]) + offset;
      for (int j = 0; j < sample_seq_len; ++j) {
        result[j * 2] = *(b0 + j);
        result[(j * 2) + 1] = *(b1 + j);
      }
      offset += sample_seq_len;

      pushResult(&((*sm)[i]), result);
    }
  };

  void applyMask(TInputDataType *ptr, int seq_len, int offset) {

    // add the y offset
    ptr += offset * packed_seq_len;

    // create the mask for the first line
    for (int i = offset; i < seq_len + offset; ++i)
      ptr[i] = 1;

    // replicate in the y direction
    for (int i = 1; i < seq_len; ++i) {
      TInputDataType *src = ptr + offset;
      TInputDataType *dst = src + packed_seq_len * i;
      memcpy(dst, src, seq_len * sizeof(TInputDataType));
    }
  }

  // function pointer hooks
  preprocessSamplesPtr pps_ptr;
  configureWorkloadPtr cw_ptr;
  postprocessResultsPtr ppr_ptr;

  // cached config
  unsigned int datasource_seq_len;
  unsigned int packed_seq_len;

  // zero buffer to blank data
  std::vector<TInputDataType> _zero_buffer;

  // handle to config
  const IConfig *_config;
};
} // namespace KRAI

#endif // BERT_MODEL_IMPL_H
