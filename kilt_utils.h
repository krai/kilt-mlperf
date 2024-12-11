#pragma once

#include <cstdint>
#include <type_traits>

#include "query_sample.h"

namespace kilt_utils {

template <typename TSampleType>
inline mlperf::QuerySample GetSample(const TSampleType& sample) {
  if constexpr (std::is_same_v<TSampleType, std::pair<mlperf::QuerySample, int>>) {  // SizedSample
    return sample.first;
  } else if constexpr (std::is_same_v<TSampleType, mlperf::QuerySample>) {
    return sample;
  }
  throw "unreachable";
}

template <typename TSampleType>
inline int32_t GetSampleID(const TSampleType& sample) {
  return GetSample(sample).id;
}

};  // namespace kilt_utils
