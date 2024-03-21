//
// MIT License
//
// Copyright (c) 2023 - 2024 Krai Ltd
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
    {"KILT_MODEL_NAME", "kilt_model_name"},
    {"KILT_MODEL_INPUT_COUNT", "kilt_input_count"},
    {"KILT_MODEL_OUTPUT_COUNT", "kilt_output_count"},
    {"KILT_MODEL_INPUT_FORMAT", "kilt_input_format"},
    {"KILT_MODEL_OUTPUT_FORMAT", "kilt_output_format"},
    {"KILT_MODEL_BATCH_SIZE", "kilt_model_batch_size"},
    {"KILT_MODEL_ROOT", "kilt_model_root"},

    // model BERT
    {"KILT_MODEL_BERT_SEQ_LENGTH", "kilt_model_seq_length"},
    {"KILT_MODEL_BERT_VARIANT", "kilt_model_bert_variant"},

    // model Object Detection
    {"KILT_MODEL_NMS_PRIOR_BIN_PATH", "kilt_prior_bin_path"},
    {"KILT_MODEL_NMS_MAX_DETECTIONS", "kilt_model_max_detections"},
    {"KILT_MODEL_NMS_DISABLE", "kilt_model_disable_nms"},

    // model GPTJ
    {"KILT_MODEL_GPTJ_BEAM_WIDTH", "kilt_beam_width"},
    {"KILT_MODEL_GPTJ_MAX_INPUT_LENGTH", "kilt_max_input_length"},
    {"KILT_MODEL_GPTJ_MAX_OUTPUT_LENGTH", "kilt_max_output_length"},
    {"KILT_MODEL_GPTJ_MIN_OUTPUT_LENGTH", "kilt_min_output_length"},
    {"KILT_MODEL_GPTJ_INPUT_TENSOR_NAMES", "input_tensor_names"},
    {"KILT_MODEL_GPTJ_OUTPUT_TENSOR_NAMES", "output_tensor_names"},
    {"KILT_MODEL_GPTJ_VARIANT", "kilt_model_gptj_variant"},

    // dataset SQUAD
    {"KILT_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH",
     "dataset_squad_tokenized_max_seq_length"},
    {"KILT_DATASET_SQUAD_TOKENIZED_ROOT", "dataset_squad_tokenized_root"},
    {"KILT_DATASET_SQUAD_TOKENIZED_INPUT_IDS",
     "dataset_squad_tokenized_input_ids"},
    {"KILT_DATASET_SQUAD_TOKENIZED_INPUT_MASK",
     "dataset_squad_tokenized_input_mask"},
    {"KILT_DATASET_SQUAD_TOKENIZED_SEGMENT_IDS",
     "dataset_squad_tokenized_segment_ids"},
    {"KILT_DATASET_SQUAD_TOKENIZED_MAX_SEQ_LENGTH",
     "dataset_squad_tokenized_max_seq_length"},

    // dataset IMAGENET
    {"KILT_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE",
     "dataset_imagenet_preprocessed_input_square_side"},
    {"KILT_DATASET_IMAGENET_HAS_BACKGROUND_CLASS",
     "ml_model_has_background_class"},
    {"KILT_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF",
     "dataset_imagenet_preprocessed_subset_fof"},
    {"KILT_DATASET_IMAGENET_PREPROCESSED_DIR",
     "dataset_imagenet_preprocessed_dir"},

    // dataset COCO / OPENIMAGES
    {"KILT_DATASET_OBJECT_DETECTION_IMAGE_HEIGHT", "ml_model_image_height"},
    {"KILT_DATASET_OBJECT_DETECTION_IMAGE_WIDTH", "ml_model_image_width"},
    {"KILT_DATASET_OBJECT_DETECTION_IMAGE_CHANNELS", "ml_model_image_channels"},
    {"KILT_DATASET_OBJECT_DETECTION_PREPROCESSED_DIR",
     "kilt_object_detection_preprocessed_dir"},
    {"KILT_DATASET_OBJECT_DETECTION_PREPROCESSED_SUBSET_FOF",
     "kilt_object_detection_preprocessed_subset_fof"},

    // dataset GPTJ / CNN DAILY MAIL
    {"KILT_DATASET_GPTJ_PREPROCESSED_ROOT", "kilt_cnndm_preprocessed_dir"},
    {"KILT_DATASET_GPTJ_PREPROCESSED_ATTENTION_MASKS", "kilt_cnndm_preprocessed_attention_masks"},
    {"KILT_DATASET_GPTJ_PREPROCESSED_INPUT_IDS_PADDED", "kilt_cnndm_preprocessed_input_ids_padded"},
    {"KILT_DATASET_GPTJ_PREPROCESSED_INPUT_LENGTHS", "kilt_cnndm_preprocessed_input_lengths"},
    {"KILT_DATASET_GPTJ_PREPROCESSED_MASKED_TOKENS", "kilt_cnndm_preprocessed_masked_tokens"},
    {"KILT_DATASET_GPTJ_MAX_SEQUENCE_LENGTH", "kilt_model_seq_length"},

    // dataset Llama2 / Openorca
    {"KILT_DATASET_LLAMA2_PREPROCESSED_ROOT", "kilt_openorca_preprocessed_dir"},
    {"KILT_DATASET_LLAMA2_PREPROCESSED_ATTENTION_MASKS", "kilt_openorca_preprocessed_attention_masks"},
    {"KILT_DATASET_LLAMA2_PREPROCESSED_INPUT_IDS_PADDED", "kilt_openorca_preprocessed_input_ids_padded"},
    {"KILT_DATASET_LLAMA2_PREPROCESSED_INPUT_LENGTHS", "kilt_openorca_preprocessed_input_lengths"},
    {"KILT_DATASET_LLAMA2_PREPROCESSED_MASKED_TOKENS", "kilt_openorca_preprocessed_masked_tokens"},
    {"KILT_DATASET_LLAMA2_MAX_SEQUENCE_LENGTH", "kilt_model_seq_length"},

 

    // dataset CNN - DAILY MAIL
    {"KILT_DATASET_CNNDM_PREPROCESSED_ROOT", "kilt_cnndm_preprocessed_dir"},
    {"KILT_DATASET_CNNDM_PREPROCESSED_ATTENTION_MASKS", "kilt_cnndm_preprocessed_attention_masks"},
    {"KILT_DATASET_CNNDM_PREPROCESSED_INPUT_IDS_PADDED", "kilt_cnndm_preprocessed_input_ids_padded"},
    {"KILT_DATASET_CNNDM_PREPROCESSED_INPUT_LENGTHS", "kilt_cnndm_preprocessed_input_lengths"},
    {"KILT_DATASET_CNNDM_PREPROCESSED_MASKED_TOKENS", "kilt_cnndm_preprocessed_masked_tokens"},
    {"KILT_DATASET_CNNDM_MAX_SEQUENCE_LENGTH", "cnndm_max_seq_len"},
    {"KILT_DATASET_CNNDM_PAD_ID", "cnndm_pad_id"},
    {"KILT_DATASET_CNNDM_END_ID", "cnndm_end_id"},

    // device qaic
    {"KILT_DEVICE_QAIC_SKIP_STAGE", "kilt_device_qaic_skip_stage"},
    {"KILT_DEVICE_QAIC_QUEUE_LENGTH", "qaic_queue_length"},
    {"KILT_DEVICE_QAIC_THREADS_PER_QUEUE", "qaic_threads_per_queue"},
    {"KILT_DEVICE_QAIC_ACTIVATION_COUNT", "qaic_activation_count"},
    {"KILT_DEVICE_QAIC_MQ_DEVICE_COUNT", "kilt_device_mq_device_count"},
    {"KILT_DEVICE_QAIC_INPUT_SELECT", "qaic_input_select"},
    {"KILT_DEVICE_QAIC_SAMPLES_QUEUE_DEPTH", "kilt_device_samples_queue_depth"},
    {"KILT_DEVICE_QAIC_RINGFENCE_DRIVER", "kilt_device_ringfence_driver"},
    {"KILT_DEVICE_QAIC_SCHEDULER_YIELD_TIME",
     "kilt_device_scheduler_yield_time"},
    {"KILT_DEVICE_QAIC_ENQUEUE_YIELD_TIME", "kilt_device_enqueue_yield_time"},
    {"KILT_DEVICE_QAIC_EXECUTION_MODE", "kilt_device_execution_mode"},

    // device TensorRT
    {"KILT_DEVICE_TENSORRT_NUMBER_OF_STREAMS", "tensorrt_number_of_stream"},
    {"KILT_DEVICE_TENSORRT_BATCH_SIZE", "tensorrt_batch_size"},
    {"KILT_DEVICE_TENSORRT_INPUT_MEMORY_MANAGEMENT_STRATEGY",
     "tensorrt_input_memory_management_strategy"},
    {"KILT_DEVICE_TENSORRT_OUTPUT_MEMORY_MANAGEMENT_STRATEGY",
     "tensorrt_ouput_memory_management_strategy"},
    {"KILT_DEVICE_TENSORRT_ENGINE_SOURCE", "engine_source"},
    {"KILT_DEVICE_TENSORRT_PLUGINS_PATH", "plugins_path"},
    {"KILT_DEVICE_TENSORRT_HAS_PLUGIN", "tensorrt_has_plugin"},
    {"KILT_DEVICE_TENSORRT_PLUGIN_NAME", "tensorrt_plugin_name"},

    // device TensorRT-LLM
    {"KILT_DEVICE_TRTLLM_GPUS_PER_NODE", "trtllm_gpus_per_node"},
    {"KILT_DEVICE_TRTLLM_TENSOR_PARALLELISM", "trtllm_tensor_parallelism"},
    {"KILT_DEVICE_TRTLLM_PIPELINE_PARALLELISM", "trtllm_pipeline_parallelism"},

    // device snpe AND device onnxrt
    {"KILT_BACKEND_TYPE", "kilt_backend_type"},

    // device SNPE
    {"SNPE_PERFORMANCE_PROFILE", "snpe_performance_profile"},

    // network
    {"KILT_NETWORK_SERVER_PORT", "network_server_port"},
    {"KILT_NETWORK_SERVER_IP_ADDRESS", "network_server_ip_address"},
    {"KILT_NETWORK_NUM_SOCKETS", "network_num_sockets"},
    {"KILT_NETWORK_PAYLOAD_SIZE", "network_payload_size"},

    // kilt base
    {"KILT_VERBOSE", "verbosity"},
    {"KILT_VERBOSE_SERVER", "CK_VERBOSE_SERVER"},
    {"KILT_JSON_CONFIG", "KILT_JSON_CONFIG"},
    {"KILT_MAX_WAIT_ABS", "kilt_max_wait_abs"},
    {"KILT_SCHEDULER_YIELD_TIME", "kilt_scheduler_yield_time"},
    {"KILT_DISPATCH_YIELD_TIME", "kilt_dispatch_yield_time"},
    {"KILT_DEVICE_IDS", "kilt_device_ids"},
    {"KILT_DEVICE_CONFIG", "kilt_device_config"},
    {"KILT_DATASOURCE_CONFIG", "kilt_datasource_config"},
    {"KILT_NETWORK_UNIQUE_SERVER_ID", "kilt_unique_server_id"},
    {"KILT_DEVICE_NAME", "device"},

    // loadgen
    {"LOADGEN_BUFFER_SIZE", "loadgen_buffer_size"},
    {"LOADGEN_DATASET_SIZE", "loadgen_dataset_size"},
    {"LOADGEN_TRIGGER_COLD_RUN", "loadgen_trigger_cold_run"},
    {"LOADGEN_MLPERF_CONF", "loadgen_mlperf_conf_path"},
    {"LOADGEN_USER_CONF", "loadgen_user_conf_path"},
    {"LOADGEN_SCENARIO", "loadgen_scenario"},
    {"LOADGEN_MODE", "loadgen_mode"},

    // end of translation table
    {"TRANSLATION_TABLE_END", ""}};

const Translation *getTranslationTable() { return translation_table; }

#endif
