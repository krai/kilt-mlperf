#ifdef KILT_BENCHMARK_STANDALONE_BERT
#include "benchmarks/standalone/bert/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_BERT_SORTING
#include "benchmarks/standalone/bert_sorting/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_BERT_CLIENT
#include "benchmarks/network/bert/client/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_BERT_SERVER
#include "benchmarks/network/bert/server/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_CLASSIFICATION
#include "benchmarks/standalone/classification/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_OBJECT_DETECTION
#include "benchmarks/standalone/object-detection/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_OBJECT_DETECTION_CLIENT
#include "benchmarks/network/object-detection/client/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_OBJECT_DETECTION_SERVER
#include "benchmarks/network/object-detection/server/benchmark_impl.h"
#else
#error Benchmark Not Defined
#endif
#ifdef KILT_DEVICE_QAIC
#include "devices/qaic/device.h"
#elif KILT_DEVICE_TENSORRT
#include "devices/tensorrt/device.h"
#elif KILT_DEVICE_ONNXRT
#include "devices/onnx/device.h"
#elif KILT_DEVICE_SNPE
#include "devices/snpe/device.h"
#elif KILT_DEVICE_DUMMY
#include "devices/dummy/device.h"
#elif KILT_DEVICE_NONE
#else
#endif
