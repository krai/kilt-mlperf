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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.POSSIBILITY OF SUCH DAMAGE.
//


#ifndef CONFIG_JSON_TOOLS_H
#define CONFIG_JSON_TOOLS_H

#include "cJSON.h"

static cJSON *cjson_object = nullptr;

char *load_json_file(char *filename) {
  void *buffer = 0;
  long length;
  FILE *f = fopen(filename, "rb");

  if (f) {
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = malloc(length);
    if (buffer) {
      fread(buffer, 1, length, f);
    }
    fclose(f);
  }

  return reinterpret_cast<char *>(buffer);
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

const char *getconfig(const char *name) {

  cJSON *j = cJSON_GetObjectItemCaseSensitive(getjSON(), name);

  if (cJSON_IsString(j) && (j->valuestring != NULL)) {
    std::cout << "CFG: " << name << " = " << j->valuestring << std::endl;
    return j->valuestring;
  }
  return nullptr;
}

//--------------------------------------------------------------------
inline char *getenv_c(const char *s) {
  return const_cast<char *>(getconfig(s));
}

/// Load mandatory string value from the environment.
inline std::string getenv_s(const std::string &name) {
  const char *value = getconfig(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return std::string(value);
}

inline std::string getenv_opt_s(const std::string &name,
                                const std::string default_value) {
  const char *value = getconfig(name.c_str());
  if (!value)
    return default_value;
  else
    return std::string(value);
}

/// Load mandatory integer value from the environment.
inline int getenv_i(const std::string &name) {
  const char *value = getconfig(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return atoi(value);
}

/// Load mandatory float value from the environment.
inline float getenv_f(const std::string &name) {
  const char *value = getconfig(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return atof(value);
}

/// Load an optional boolean value from the environment.
inline bool getenv_b(const char *name) {
  std::string value = getconfig(name);

  return (value == "YES" || value == "yes" || value == "ON" || value == "on" ||
          value == "1");
}

inline std::string alter_str(std::string a, std::string b) {
  return a != "" ? a : b;
};
inline std::string alter_str(char *a, std::string b) {
  return a != nullptr ? a : b;
};
inline std::string alter_str(char *a, char *b) {
  return a != nullptr ? a : b;
};
inline int alter_str_i(char *a, int b) {
  return a != nullptr ? std::atoi(a) : b;
};
inline int alter_str_i(char *a, char *b) {
  return std::atoi(a != nullptr ? a : b);
};
inline int alter_str_i(std::string a, std::string b) {
  return std::atoi(a != "" ? a.c_str() : b.c_str());
};
// inline float alter_str_f(std::string a, std::string b) {
//  return std::atof(a != "" ? a.c_str() : b.c_str());
//};
inline float alter_str_f(const char *a, const char *b) {
  return std::atof(a != nullptr ? a : b);
};

/// Dummy `sprintf` like formatting function using std::string.
/// It uses buffer of fixed length so can't be used in any cases,
/// generally use it for short messages with numeric arguments.
template <typename... Args>
inline std::string format(const char *str, Args... args) {
  char buf[1024];
  sprintf(buf, str, args...);
  return std::string(buf);
};

#endif // CONFIG_JSON_TOOLS_H
