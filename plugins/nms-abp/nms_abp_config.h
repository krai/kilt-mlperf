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

#include <vector>

class R34_Params {
public:
  const int NUM_CLASSES = 81;
  const int MAX_BOXES_PER_CLASS = 100;
  const int TOTAL_NUM_BOXES = 15130;

  const int DATA_LENGTH_LOC = 60520;
  const int DATA_LENGTH_CONF = 1225530;

  const int BOX_ITR_0 = 0;
  const int BOX_ITR_1 = (TOTAL_NUM_BOXES * 1);
  const int BOX_ITR_2 = (TOTAL_NUM_BOXES * 2);
  const int BOX_ITR_3 = (TOTAL_NUM_BOXES * 3);

  const int OFFSET_CONF = 15130;
  const int BOXES_INDEX = 0;
  const int CLASSES_INDEX = 1;

  const int CLASSES_OFFSET = 1;

  const float LOC_OFFSET = 0.0f;
  const float LOC_SCALE = 0.134f;
  const float CONF_OFFSET = 0.0f;
  const float CONF_SCALE = 1.0f;

  const float CLASS_THRESHOLD = 0.05f;
  const int CLASS_THRESHOLD_UINT8 = 0; // fixme
  const int CLASS_THRESHOLD_FP16 = 10854;
  const float NMS_THRESHOLD = 0.5f;
  const int KILT_MODEL_NMS_MAX_DETECTIONS_PER_IMAGE = 600;
  const int KILT_MODEL_NMS_MAX_DETECTIONS_PER_CLASS = 100;

  const char *priorName = "R34_priors.bin";
  const bool MAP_CLASSES = true;
  const bool PREPROCESS_PRIOR = false;
  std::vector<float> variance = {0.1, 0.2};
  std::vector<float> class_map = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37,
      38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
      56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76,
      77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
};

class MV1_Params {
public:
  const int NUM_CLASSES = 91;
  const int MAX_BOXES_PER_CLASS = 100;
  const int TOTAL_NUM_BOXES = 1917;

  const int DATA_LENGTH_LOC = 7668;
  const int DATA_LENGTH_CONF = 17447;

  const int BOX_ITR_0 = 0;
  const int BOX_ITR_1 = 1;
  const int BOX_ITR_2 = 2;
  const int BOX_ITR_3 = 3;

  const int OFFSET_CONF = 1;
  const int BOXES_INDEX = 1;
  const int CLASSES_INDEX = 0;

  const int CLASSES_OFFSET = 1;

  const float LOC_OFFSET = 0.0f;
  const float LOC_SCALE = 0.144255146f;
  const float CONF_OFFSET = -128.0f;
  const float CONF_SCALE = 0.00392156886f;

  const float CLASS_THRESHOLD = 0.3f;
  const int CLASS_THRESHOLD_UINT8 = 76;
  const int CLASS_THRESHOLD_FP16 = 0; // fixme
  const float NMS_THRESHOLD = 0.45f;
  const int KILT_MODEL_NMS_MAX_DETECTIONS_PER_IMAGE = 100;
  const int KILT_MODEL_NMS_MAX_DETECTIONS_PER_CLASS = 100;

  const char *priorName = "MV1_priors.bin";
  const bool MAP_CLASSES = false;
  const bool PREPROCESS_PRIOR = false;
  std::vector<float> variance = {};
  std::vector<float> class_map = {};
};

class RX50_Params {
public:
  const int NUM_CLASSES = 264;
  const int MAX_BOXES_PER_CLASS = 200;
  const int TOTAL_NUM_BOXES = 120087;

  const int DATA_LENGTH_LOC = 480348;
  const int DATA_LENGTH_CONF = 32183316;

  const int BOX_ITR_0 = 0;
  const int BOX_ITR_1 = (TOTAL_NUM_BOXES * 1);
  const int BOX_ITR_2 = (TOTAL_NUM_BOXES * 2);
  const int BOX_ITR_3 = (TOTAL_NUM_BOXES * 3);

  const int OFFSET_CONF = 120087;

#ifdef SDK_1_11_X
  const int CLASSES_INDEX = 5;
  const int BOXES_INDEX = 10;
  const int TOPK_INDEX = 0;
#else
  const int CLASSES_INDEX = 0;
  const int BOXES_INDEX = 5;
  const int TOPK_INDEX = 10;
#endif

  const int CLASSES_OFFSET = 0;

  const int OUTPUT_LEVELS = 5;
  const int OUTPUT_BOXES_PER_LEVEL = 1000;
  const int OUTPUT_DELTAS[5] = {0, 90000, 22500, 5625, 1521};

#ifndef LOC_OFFSET
  const float LOC_OFFSET = 25.0f;
#endif
#ifndef LOC_SCALE
  const float LOC_SCALE = 0.01684683f;
#endif
#ifndef CONF_OFFSET
  const float CONF_OFFSET = -128.0f;
#endif
#ifndef CONF_SCALE
  const float CONF_SCALE = 0.00388179976f;
#endif

  const float CLASS_THRESHOLD = 0.05f;
  const int CLASS_THRESHOLD_UINT8 = 5;
  const int CLASS_THRESHOLD_FP16 = 10854;
  const float NMS_THRESHOLD = 0.5f;
  const int KILT_MODEL_NMS_MAX_DETECTIONS_PER_IMAGE = 1000;
  const int KILT_MODEL_NMS_MAX_DETECTIONS_PER_CLASS = 1000;

  //   const float BOX_SCALE = 0.00125f;
  const float BOX_SCALE = 800.0f;

  const char *priorName = "retinanet_priors.bin";
  const bool MAP_CLASSES = false;
  const bool PREPROCESS_PRIOR = true;
  std::vector<float> variance = {};
  std::vector<float> class_map = {};
};

#define CONVERT_TO_INT8(x) ((int8_t)((int16_t)x - 128))
#define CLASS_POSITION 6
#define SCORE_POSITION 5
#define NUM_COORDINATES 4
#define CONVERT_UINT8_FP32(x, offset, scale)                                   \
  ((CONVERT_TO_INT8(x) - offset) * scale)

#define CONVERT_INT8_FP32(x, offset, scale) ((x - offset) * scale)

using bbox = std::vector<float>;
