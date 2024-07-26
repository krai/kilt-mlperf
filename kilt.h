#include "kilt_utils.h"

#ifdef KILT_BENCHMARK_STANDALONE_BERT
#define STANDALONE
#include "benchmarks/standalone/bert/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_BERT_SORTING
#define STANDALONE
#include "benchmarks/standalone/bert_sorting/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_BERT_CLIENT
#include "benchmarks/network/bert/client/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_BERT_SERVER
#include "benchmarks/network/bert/server/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_CLASSIFICATION
#define STANDALONE
#include "benchmarks/standalone/classification/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_CLASSIFICATION_CLIENT
#include "benchmarks/network/classification/client/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_CLASSIFICATION_SERVER
#include "benchmarks/network/classification/server/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_OBJECT_DETECTION
#define STANDALONE
#include "benchmarks/standalone/object-detection/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_OBJECT_DETECTION_CLIENT
#include "benchmarks/network/object-detection/client/benchmark_impl.h"
#elif KILT_BENCHMARK_NETWORK_OBJECT_DETECTION_SERVER
#include "benchmarks/network/object-detection/server/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_GPTJ
#define STANDALONE
#include "benchmarks/standalone/gptj/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_LLAMA2
#define STANDALONE
#include "benchmarks/standalone/llama2/benchmark_impl.h"
#elif KILT_BENCHMARK_STANDALONE_GPTJ_TRTLLM
#define STANDALONE
#include "benchmarks/standalone/gptj_trtllm/benchmark_impl.h"
#elif KILT_BENCHMARK_EXTERNAL
#include "benchmark_impl.h"
#else
#error Benchmark Not Defined
#endif
#if KILT_DEVICE_ONNXRT
#include "devices/onnx/device.h"
#elif KILT_DEVICE_DUMMY
#include "devices/dummy/device.h"
#elif KILT_DEVICE_EXTERNAL
#include "device.h"
#elif KILT_DEVICE_NONE
#else
#error Device Not Defined!
#endif
