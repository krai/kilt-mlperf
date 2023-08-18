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

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "nms_abp_config.h"

#include "fp16.h"
template <typename Loc, typename Conf, typename MParams> class NMS_ABP {
  std::string binPath;

public:
  std::string priorName;
  float *priorTensor;
  MParams modelParams;
  NMS_ABP(const std::string &path) {
    binPath = path;
    if (binPath == "")
      binPath = ".";
    readPriors();
    if (modelParams.PREPROCESS_PRIOR)
      preprocessPrior();
  }
  ~NMS_ABP() { delete priorTensor; };
  void preprocessPrior() {

    for (uint32_t i = 0; i < modelParams.TOTAL_NUM_BOXES; i++) {

      float *prior = &priorTensor[NUM_COORDINATES * i];
      float w = prior[2] - prior[0];
      float h = prior[3] - prior[1];
      float cent_x = prior[0] + 0.5f * w;
      float cent_y = prior[1] + 0.5f * h;

      prior[0] = w;
      prior[1] = h;
      prior[2] = cent_x;
      prior[3] = cent_y;
    }
  }
  float *read(std::string priorFilename, uint32_t tensorLength) {
    std::ifstream fs(binPath + "/" + priorFilename, std::ifstream::binary);
    fs.seekg(0, std::ios::end);
    uint32_t fileSize = fs.tellg();
    fs.seekg(0, std::ios::beg);

    if (tensorLength != fileSize) {
      std::cerr << "Invalid input: " << priorFilename << std::endl;
      std::cerr << "Length mismatch: "
                << " Tensor Size: " << tensorLength
                << ",\t File Size: " << fileSize << std::endl;
      std::exit(1);
    }
    float *priorData = new float[tensorLength];
    fs.read((char *)priorData, tensorLength);
    fs.close();
    return priorData;
  }

public:
  void readPriors() {
    priorTensor =
        read(modelParams.priorName,
             modelParams.TOTAL_NUM_BOXES * NUM_COORDINATES * sizeof(float));
  }

  void anchorBoxProcessing(const Loc *const locTensor,
                           const Conf *const confTensor,
                           std::vector<bbox> &selectedAll, const float idx) {

    const Conf *confPtr = confTensor;
    const Loc *locPtr = locTensor;

    float const *priorPtr = priorTensor;
#if defined(MODEL_R34)

    for (uint32_t ci = modelParams.CLASSES_OFFSET; ci < modelParams.NUM_CLASSES;
         ci++) {

      uint32_t confItr = ci * modelParams.OFFSET_CONF;
      std::vector<bbox> result;
      std::vector<bbox> selected;
      confPtr = confTensor;
      locPtr = locTensor;
      priorPtr = priorTensor;
#ifdef __amd64__
      for (uint32_t bi = 0; bi < modelParams.TOTAL_NUM_BOXES;
#else
      for (uint32_t bi = 0; bi < 15130;
#endif
           ++bi, confPtr++, locPtr++, priorPtr++) {

        Conf confidence = confPtr[confItr];
        if (confidence < 10854)
          continue;
        // if (!above_Class_Threshold(confidence)) continue;
        float cf = get_Score_Val(confidence);
        bbox cBox = {get_Loc_Val(locPtr[modelParams.BOX_ITR_0]),
                     get_Loc_Val(locPtr[modelParams.BOX_ITR_1]),
                     get_Loc_Val(locPtr[modelParams.BOX_ITR_2]),
                     get_Loc_Val(locPtr[modelParams.BOX_ITR_3])};
        if (modelParams.variance.data() != NULL)
          cBox =
              decodeLocationTensor(cBox, priorPtr, modelParams.variance.data());
        else
          cBox = decodeLocationTensor(cBox, priorPtr);
        result.emplace_back(std::initializer_list<float>{
            idx, cBox[1], cBox[0], cBox[3], cBox[2], cf, (float)ci});
      }

      if (result.size()) {
        NMS(result, modelParams.NMS_THRESHOLD, modelParams.MAX_BOXES_PER_CLASS,
            selected, selectedAll, modelParams.class_map);
      }
    }
#else // MV1 and RX50
    std::vector<bbox> result[modelParams.NUM_CLASSES];
    std::vector<bbox> selected[modelParams.NUM_CLASSES];
    for (uint32_t bi = 0; bi < modelParams.TOTAL_NUM_BOXES;
         bi++, locPtr += 4, priorPtr += 4) {
      uint32_t confItr = bi * modelParams.NUM_CLASSES;
      for (uint32_t ci = modelParams.CLASSES_OFFSET;
           ci < modelParams.NUM_CLASSES; ci++) {

        Conf confidence = confPtr[confItr + ci];
        if (!above_Class_Threshold(confidence))
          continue;
        float cf = get_Score_Val(confidence);
        bbox cBox = {get_Loc_Val(locPtr[0]), get_Loc_Val(locPtr[1]),
                     get_Loc_Val(locPtr[2]), get_Loc_Val(locPtr[3])};
        if (modelParams.variance.data() != NULL)
          cBox =
              decodeLocationTensor(cBox, priorPtr, modelParams.variance.data());
        else
          cBox = decodeLocationTensor(cBox, priorPtr);
        result[ci].emplace_back(std::initializer_list<float>{
            idx, cBox[1], cBox[0], cBox[3], cBox[2], cf, (float)ci});
      }
    }

    for (uint32_t ci = modelParams.CLASSES_OFFSET; ci < modelParams.NUM_CLASSES;
         ci++) {
      if (result[ci].size()) {
        NMS(result[ci], modelParams.NMS_THRESHOLD,
            modelParams.MAX_BOXES_PER_CLASS, selected[ci], selectedAll,
            modelParams.class_map);
      }
    }
#endif
    int middle = selectedAll.size();
    if (middle > modelParams.KILT_MODEL_NMS_MAX_DETECTIONS_PER_IMAGE) {
      middle = modelParams.KILT_MODEL_NMS_MAX_DETECTIONS_PER_IMAGE;
    }
    std::partial_sort(selectedAll.begin(), selectedAll.begin() + middle,
                      selectedAll.end(), [](const bbox &a, const bbox &b) {
                        return a[SCORE_POSITION] > b[SCORE_POSITION];
                      });
  }

#if defined(MODEL_RX50)
  void anchorBoxProcessing(const Loc **const locTensor,
                           const Conf **const confTensor,
                           const uint64_t **const topkTensor,
                           std::vector<bbox> &selectedAll, const float idx) {

    std::vector<bbox> result[modelParams.NUM_CLASSES];
    std::vector<bbox> selected[modelParams.NUM_CLASSES];

    uint32_t prior_offset = 0;

    for (uint32_t gi = 0; gi < modelParams.OUTPUT_LEVELS; ++gi) {
      prior_offset += modelParams.OUTPUT_DELTAS[gi];

      const Loc *locPtr = locTensor[gi];

      for (uint32_t bi = 0; bi < modelParams.OUTPUT_BOXES_PER_LEVEL;
           ++bi, locPtr += 4) {

        bbox cBox = {get_Loc_Val(locPtr[0]), get_Loc_Val(locPtr[1]),
                     get_Loc_Val(locPtr[2]), get_Loc_Val(locPtr[3])};

        uint32_t cls = (uint32_t)topkTensor[gi][bi] % modelParams.NUM_CLASSES;
        uint32_t off = prior_offset +
                       (uint32_t)topkTensor[gi][bi] / modelParams.NUM_CLASSES;
        float cf = get_Score_Val(confTensor[gi][bi]);
        if (!above_Class_Threshold(cf))
          continue;

        if (modelParams.variance.data() != NULL)
          cBox = decodeLocationTensor(cBox, &priorTensor[off * 4],
                                      modelParams.variance.data());
        else
          cBox = decodeLocationTensor(cBox, &priorTensor[off * 4]);

        result[cls].emplace_back(std::initializer_list<float>{
            idx, cBox[1], cBox[0], cBox[3], cBox[2], cf, (float)cls});
      }
    }

    for (uint32_t ci = modelParams.CLASSES_OFFSET; ci < modelParams.NUM_CLASSES;
         ci++) {
      if (result[ci].size()) {
        NMS(result[ci], modelParams.NMS_THRESHOLD,
            modelParams.MAX_BOXES_PER_CLASS, selected[ci], selectedAll,
            modelParams.class_map);
      }
    }

    int middle = selectedAll.size();
    if (middle > modelParams.KILT_MODEL_NMS_MAX_DETECTIONS_PER_IMAGE) {
      middle = modelParams.KILT_MODEL_NMS_MAX_DETECTIONS_PER_IMAGE;
    }
    std::partial_sort(selectedAll.begin(), selectedAll.begin() + middle,
                      selectedAll.end(), [](const bbox &a, const bbox &b) {
                        return a[SCORE_POSITION] > b[SCORE_POSITION];
                      });

    for (uint32_t b = 0; b < selectedAll.size(); ++b) {
      postproc(selectedAll[b][1]);
      postproc(selectedAll[b][2]);
      postproc(selectedAll[b][3]);
      postproc(selectedAll[b][4]);
    }
  }
#endif

  inline void postproc(float &box) {
    box /= modelParams.BOX_SCALE;
    if (box < 0.0f)
      box = 0.0f;
    if (box > 1.0f)
      box = 1.0f;
  }

  inline Conf above_Class_Threshold(uint8_t score) {
    return score > modelParams.CLASS_THRESHOLD_UINT8;
  }
  inline Conf above_Class_Threshold(int8_t score) {
    return score > modelParams.CLASS_THRESHOLD_UINT8;
  }
  inline Conf above_Class_Threshold(uint16_t score) {
    //      return score > modelParams.CLASS_THRESHOLD_FP16;
    return fp16_ieee_to_fp32_value(score) > modelParams.CLASS_THRESHOLD;
  }
  inline Conf above_Class_Threshold(float score) {
    return score > modelParams.CLASS_THRESHOLD;
  }
  inline float get_Loc_Val(uint8_t x) {
    return CONVERT_UINT8_FP32(x, modelParams.LOC_OFFSET, modelParams.LOC_SCALE);
  }
  inline float get_Loc_Val(int8_t x) {
    return CONVERT_INT8_FP32(x, modelParams.LOC_OFFSET, modelParams.LOC_SCALE);
  }
  inline float get_Loc_Val(float x) { return x; }
  inline float get_Loc_Val(uint16_t x) { return fp16_ieee_to_fp32_value(x); }
  inline float get_Score_Val(uint16_t x) { return fp16_ieee_to_fp32_value(x); }
  inline float get_Score_Val(uint8_t x) {
    return CONVERT_UINT8_FP32(x, modelParams.CONF_OFFSET,
                              modelParams.CONF_SCALE);
  }
  inline float get_Score_Val(int8_t x) {
    return CONVERT_INT8_FP32(x, modelParams.CONF_OFFSET,
                             modelParams.CONF_SCALE);
  }
  inline float get_Score_Val(float x) { return x; }
  bbox decodeLocationTensor(const bbox &loc, const float *const prior,
                            const float *const var) {

    float x = prior[modelParams.BOX_ITR_0] +
              loc[0] * var[0] * prior[modelParams.BOX_ITR_2];
    float y = prior[modelParams.BOX_ITR_1] +
              loc[1] * var[0] * prior[modelParams.BOX_ITR_3];
    float w = prior[modelParams.BOX_ITR_2] * expf(loc[2] * var[1]);
    float h = prior[modelParams.BOX_ITR_3] * expf(loc[3] * var[1]);
    x -= (w / 2.0f);
    y -= (h / 2.0f);
    w += x;
    h += y;

    return {x, y, w, h};
  }

#if defined(MODEL_RX50)
  bbox decodeLocationTensor(const bbox &loc, const float *const prior) {

    float w = prior[0];
    float h = prior[1];
    float cent_x = prior[2];
    float cent_y = prior[3];

    float dx = loc[0];
    float dy = loc[1];
    float dw = loc[2];
    float dh = loc[3];

    float pred_cent_x = dx * w + cent_x;
    float pred_cent_y = dy * h + cent_y;
    float pred_w = expf(dw) * w;
    float pred_h = expf(dh) * h;

    return {(pred_cent_x - 0.5f * pred_w), (pred_cent_y - 0.5f * pred_h),
            (pred_cent_x + 0.5f * pred_w), (pred_cent_y + 0.5f * pred_h)};
  }
#else
  bbox decodeLocationTensor(const bbox &loc, const float *const prior) {

    float w = prior[3] - prior[1];
    float h = prior[2] - prior[0];
    float cent_x = prior[1] + 0.5f * w;
    float cent_y = prior[0] + 0.5f * h;

    float dy = loc[0] / 10.0f;
    float dx = loc[1] / 10.0f;
    float dh = loc[2] / 5.0f;
    float dw = loc[3] / 5.0f;

    float pred_cent_x = dx * w + cent_x;
    float pred_cent_y = dy * h + cent_y;
    float pred_w = expf(dw) * w;
    float pred_h = expf(dh) * h;

    return {pred_cent_x - 0.5f * pred_w, pred_cent_y - 0.5f * pred_h,
            pred_cent_x + 0.5f * pred_w, pred_cent_y + 0.5f * pred_h};
  }
#endif

  template <typename A, typename B>
  void pack(const std::vector<A> &part1, const std::vector<B> &part2,
            std::vector<std::pair<A, B>> &packed) {
    assert(part1.size() == part2.size());
    for (size_t i = 0; i < part1.size(); i++) {
      packed.push_back(std::make_pair(part1[i], part2[i]));
    }
  }

  template <typename A, typename B>
  void unpack(const std::vector<std::pair<A, B>> &packed, std::vector<A> &part1,
              std::vector<B> &part2) {
    for (size_t i = 0; i < part1.size(); i++) {
      part1[i] = packed[i].first;
      part2[i] = packed[i].second;
    }
  }
#define AREA(y1, x1, y2, x2) ((y2 - y1) * (x2 - x1))
  float computeIOU(const float *box1, const float *box2) {
    float box1_y1 = box1[1], box1_x1 = box1[2], box1_y2 = box1[3],
          box1_x2 = box1[4];
    float box2_y1 = box2[1], box2_x1 = box2[2], box2_y2 = box2[3],
          box2_x2 = box2[4];

    assert(box1_y1 < box1_y2 && box1_x1 < box1_x2);
    assert(box2_y1 < box2_y2 && box2_x1 < box2_x2);

    float inter_y1 = std::max(box1_y1, box2_y1);
    float inter_x1 = std::max(box1_x1, box2_x1);
    float inter_y2 = std::min(box1_y2, box2_y2);
    float inter_x2 = std::min(box1_x2, box2_x2);

    float IOU = 0.0f;

    if ((inter_y1 < inter_y2) &&
        (inter_x1 < inter_x2)) // there is a valid intersection
    {
      float intersect = AREA(inter_y1, inter_x1, inter_y2, inter_x2);
      float total = AREA(box1_y1, box1_x1, box1_y2, box1_x2) +
                    AREA(box2_y1, box2_x1, box2_y2, box2_x2) - intersect;
      IOU = total > 0.0f ? (intersect / total) : 0.0f;
    }
    return IOU;
  }

  void insertSelected(std::vector<std::vector<float>> &selected,
                      std::vector<std::vector<float>> &selectedAll,
                      std::vector<float> &cand, const float &thres,
                      std::vector<float> &classmap) {
    for (int i = 0; i < selected.size(); i++) {
      if (computeIOU(&cand[0], &selected[i][0]) > thres) {
        return;
      }
    }
    cand[SCORE_POSITION] = cand[5];
    if (modelParams.MAP_CLASSES)
      cand[CLASS_POSITION] = classmap[cand[CLASS_POSITION]];
    selected.push_back(cand);
    selectedAll.push_back(cand);
  }

  void NMS(std::vector<std::vector<float>> &boxes, const float &thres,
           const int &max_output_size,
           std::vector<std::vector<float>> &selected,
           std::vector<std::vector<float>> &selectedAll,
           std::vector<float> &classmap) {

    std::sort(std::begin(boxes), std::end(boxes),
              [&](const auto &a, const auto &b) { return (a[5] > b[5]); });
    for (int i = 0; (i < boxes.size()) && (selected.size() < max_output_size);
         i++) {
      insertSelected(selected, selectedAll, boxes[i], thres, classmap);
    }
  }
};
