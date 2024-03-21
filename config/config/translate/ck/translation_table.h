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

#ifndef TRANSLATION_TABLE_H
#define TRANSLATION_TABLE_H

#include "../translate_tools.h"

Translation translation_table[] = {

    // model base
    {"KILT_MODEL_NAME", "ML_MODEL_MODEL_NAME"},
    {"KILT_MODEL_INPUT_COUNT", "CK_ENV_QAIC_INPUT_COUNT"},
    {"KILT_MODEL_OUTPUT_COUNT", "CK_ENV_QAIC_OUTPUT_COUNT"},
    {"KILT_MODEL_INPUT_FORMAT", "KILT_MODEL_INPUT_FORMAT"},
    {"KILT_MODEL_OUTPUT_FORMAT", "KILT_MODEL_OUTPUT_FORMAT"},
    {"KILT_MODEL_BATCH_SIZE", "CK_ENV_QAIC_MODEL_BATCH_SIZE"},

    {"KILT_MODEL_ROOT", "CK_ENV_QAIC_MODEL_ROOT"},

    // model BERT
    {"KILT_MODEL_BERT_SEQ_LENGTH", "ML_MODEL_SEQ_LENGTH"},
    {"KILT_MODEL_BERT_VARIANT", "KILT_MODEL_BERT_VARIANT"},

    // model Object Detection
    {"KILT_MODEL_NMS_PRIOR_BIN_PATH", "PRIOR_BIN_PATH"},
    {"KILT_MODEL_NMS_MAX_DETECTIONS", "CK_ENV_QAIC_MODEL_MAX_DETECTIONS"},
    {"KILT_MODEL_NMS_DISABLE", "CK_ENV_DISABLE_NMS"},

    // dataset SQUAD
    {"KILT_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH",
     "CK_ENV_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH"},
    {"KILT_DATASET_SQUAD_TOKENIZED_ROOT",
     "CK_ENV_DATASET_SQUAD_TOKENIZED_ROOT"},
    {"KILT_DATASET_SQUAD_TOKENIZED_INPUT_IDS",
     "CK_ENV_DATASET_SQUAD_TOKENIZED_INPUT_IDS"},
    {"KILT_DATASET_SQUAD_TOKENIZED_INPUT_MASK",
     "CK_ENV_DATASET_SQUAD_TOKENIZED_INPUT_MASK"},
    {"KILT_DATASET_SQUAD_TOKENIZED_SEGMENT_IDS",
     "CK_ENV_DATASET_SQUAD_TOKENIZED_SEGMENT_IDS"},
    {"KILT_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH",
     "CK_ENV_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH"},

    // dataset IMAGENET
    {"KILT_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE",
     "CK_ENV_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE"},
    {"KILT_DATASET_IMAGENET_HAS_BACKGROUND_CLASS",
     "ML_MODEL_HAS_BACKGROUND_CLASS"},
    {"KILT_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF",
     "CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF"},
    {"KILT_DATASET_IMAGENET_PREPROCESSED_DIR",
     "CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR"},

    // dataset COCO / OPENIMAGES
    {"KILT_DATASET_OBJECT_DETECTION_IMAGE_HEIGHT", "ML_MODEL_IMAGE_HEIGHT"},
    {"KILT_DATASET_OBJECT_DETECTION_IMAGE_WIDTH", "ML_MODEL_IMAGE_WIDTH"},
    {"KILT_DATASET_OBJECT_DETECTION_IMAGE_CHANNELS", "ML_MODEL_IMAGE_CHANNELS"},
    {"KILT_DATASET_OBJECT_DETECTION_PREPROCESSED_DIR",
     "CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR"},
    {"KILT_DATASET_OBJECT_DETECTION_PREPROCESSED_SUBSET_FOF",
     "CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF"},

    // device qaic
    {"KILT_DEVICE_QAIC_SKIP_STAGE", "CK_ENV_QAIC_SKIP_STAGE"},
    {"KILT_DEVICE_QAIC_QUEUE_LENGTH", "CK_ENV_QAIC_QUEUE_LENGTH"},
    {"KILT_DEVICE_QAIC_THREADS_PER_QUEUE", "CK_ENV_QAIC_THREADS_PER_QUEUE"},
    {"KILT_DEVICE_QAIC_ACTIVATION_COUNT", "CK_ENV_QAIC_ACTIVATION_COUNT"},
    {"KILT_DEVICE_QAIC_INPUT_SELECT", "CK_ENV_QAIC_INPUT_SELECT"},
    {"KILT_DEVICE_QAIC_SAMPLES_QUEUE_DEPTH",
     "KILT_DEVICE_QAIC_SAMPLES_QUEUE_DEPTH"},
    {"KILT_DEVICE_QAIC_RINGFENCE_DRIVER", "KILT_DEVICE_QAIC_RINGFENCE_DRIVER"},
    {"KILT_DEVICE_QAIC_SCHEDULER_YIELD_TIME",
     "KILT_DEVICE_SCHEDULER_YIELD_TIME"},
    {"KILT_DEVICE_QAIC_ENQUEUE_YIELD_TIME", "KILT_DEVICE_ENQUEUE_YIELD_TIME"},

    // network
    {"KILT_NETWORK_SERVER_PORT", "NETWORK_SERVER_PORT"},
    {"KILT_NETWORK_SERVER_IP_ADDRESS", "NETWORK_SERVER_IP_ADDRESS"},
    {"KILT_NETWORK_NUM_SOCKETS", "NETWORK_NUM_SOCKETS"},
    {"KILT_NETWORK_PAYLOAD_SIZE", "KILT_NETWORK_PAYLOAD_SIZE"},

    // kilt base
    {"KILT_VERBOSE", "CK_VERBOSE"},
    {"KILT_VERBOSE_SERVER", "CK_VERBOSE_SERVER"},
    {"KILT_JSON_CONFIG", "KILT_JSON_CONFIG"},
    {"KILT_MAX_WAIT_ABS", "CK_ENV_QAIC_MAX_WAIT_ABS"},
    {"KILT_SCHEDULER_YIELD_TIME", "KILT_SCHEDULER_YIELD_TIME"},
    {"KILT_DISPATCH_YIELD_TIME", "KILT_DISPATCH_YIELD_TIME"},
    {"KILT_DEVICE_IDS", "CK_ENV_QAIC_DEVICE_IDS"},
    {"KILT_DEVICE_CONFIG", "CK_ENV_QAIC_DEVICE_CONFIG"},
    {"KILT_DATASOURCE_CONFIG", "CK_ENV_QAIC_DATASOURCE_CONFIG"},
    {"KILT_NETWORK_UNIQUE_SERVER_ID", "CK_ENV_UNIQUE_SERVER_ID"},
    {"KILT_DEVICE_QAIC_LOOPBACK", "KILT_DEVICE_QAIC_LOOPBACK"},

    // loadgen
    {"LOADGEN_BUFFER_SIZE", "CK_LOADGEN_BUFFER_SIZE"},
    {"LOADGEN_DATASET_SIZE", "CK_LOADGEN_DATASET_SIZE"},
    {"LOADGEN_TRIGGER_COLD_RUN", "CK_LOADGEN_TRIGGER_COLD_RUN"},
    {"LOADGEN_MLPERF_CONF", "CK_ENV_MLPERF_INFERENCE_MLPERF_CONF"},
    {"LOADGEN_USER_CONF", "CK_LOADGEN_USER_CONF"},
    {"LOADGEN_SCENARIO", "CK_LOADGEN_SCENARIO"},
    {"LOADGEN_MODE", "CK_LOADGEN_MODE"},

    // end of translation table
    {"TRANSLATION_TABLE_END", ""}};

const Translation *getTranslationTable() { return translation_table; }

#endif