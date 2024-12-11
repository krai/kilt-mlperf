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

#ifndef READ_CONFIG_H
#define READ_CONFIG_H

#include <assert.h>

#include "cJSON.h"
#include "config/translate/kilt_translate.h"
#include "string"

static cJSON *cjson_object = nullptr;

char *load_json_file(const char *filename) {
  void *buffer = 0;
  long length;
  FILE *f = fopen(filename, "rb");

  if (f) {
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = malloc(length);
    if (buffer) {
      assert(fread(buffer, 1, length, f));
    }
    fclose(f);
  }

  return reinterpret_cast<char *>(buffer);
}

void setJSONConfig(const char *filename) {
  char *input_json_string = load_json_file(filename);

  cjson_object = cJSON_Parse(input_json_string);

  if (cjson_object == nullptr) {
    const char *error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != nullptr) {
      fprintf(stderr, "Error before: %s\n", error_ptr);
      exit(1);
    }
  }
}

cJSON *getjSON() {

  if (cjson_object == nullptr) {

    char *input_json_string = load_json_file(getenv("KILT_JSON_CONFIG"));

    cjson_object = cJSON_Parse(input_json_string);

    if (cjson_object == nullptr) {
      const char *error_ptr = cJSON_GetErrorPtr();
      if (error_ptr != nullptr) {
        fprintf(stderr, "Error before: %s\n", error_ptr);
        exit(1);
      }
    }
  }

  return cjson_object;
}

const char *getconfig_c(const char *name) {

  static std::string config_str;

  cJSON *j = cJSON_GetObjectItemCaseSensitive(
      getjSON(), TranslationTable::getTranslation(name).c_str());

  if (cJSON_IsString(j) && (j->valuestring != NULL)) {
    std::cout << "CFG: " << name << " = " << j->valuestring << std::endl;
    return j->valuestring;
  } else if (cJSON_IsNumber(j)) {
    std::cout << "CFG: " << name << " = " << j->valueint << std::endl;
    config_str = std::to_string(j->valueint);
    return config_str.c_str();
  } else if (cJSON_IsBool(j)) {
    std::cout << "CFG: " << name << " = " << j->valueint << std::endl;
    return (j->valueint ? "true" : "false");
  }
  return nullptr;
}

#endif // READ_CONFIG_H
