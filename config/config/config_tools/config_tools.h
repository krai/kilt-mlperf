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

#ifndef CONFIG_TOOLS_H
#define CONFIG_TOOLS_H

#ifdef KILT_CONFIG_FROM_JSON
#include "json/read_config.h"
#elif KILT_CONFIG_FROM_ENV
#include "env/read_config.h"
#else
#error Config reader backend not defined.
#endif

#include "config/translate/kilt_translate.h"

#include "string"

/// Load mandatory string value from the environment.
inline std::string getconfig_s(const std::string &name) {
  const char *value = getconfig_c(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return std::string(value);
}

inline std::string getconfig_opt_s(const std::string &name,
                                   const std::string default_value) {
  const char *value = getconfig_c(name.c_str());
  if (!value)
    return default_value;
  else
    return std::string(value);
}

inline bool getconfig_opt_b(const std::string &name, bool default_value) {

  const char *value = getconfig_c(name.c_str());
  if (!value)
    return default_value;
  else {
    return (!strcmp(value, "YES") || !strcmp(value, "yes") ||
            !strcmp(value, "ON") || !strcmp(value, "on") ||
            !strcmp(value, "1") || !strcmp(value, "true"));
  }
}

/// Load mandatory integer value from the environment.
inline int getconfig_i(const std::string &name) {
  const char *value = getconfig_c(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return atoi(value);
}

/// Load mandatory float value from the environment.
inline float getconfig_f(const std::string &name) {
  const char *value = getconfig_c(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return atof(value);
}

/// Load an optional boolean value from the environment.
inline bool getconfig_b(const char *name) {
  std::string value = getconfig_c(name);

  return (value == "YES" || value == "yes" || value == "ON" || value == "on" ||
          value == "true" || value == "1");
}

inline std::string alter_str(std::string a, std::string b) {
  return a != "" ? a : b;
};
inline std::string alter_str(const char *a, std::string b) {
  return a != nullptr ? a : b;
};
inline std::string alter_str(const char *a, const char *b) {
  return a != nullptr ? a : b;
};
inline int alter_str_i(const char *a, int b) {
  return a != nullptr ? std::atoi(a) : b;
};
inline int alter_str_i(const char *a, const char *b) {
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

#endif // CONFIG_TOOLS_H
