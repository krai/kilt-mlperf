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

#ifndef TRANSLATE_TOOLS_H
#define TRANSLATE_TOOLS_H

#include <cstring>
#include <iostream>
#include <map>

struct Translation {
  const char *key;
  const char *value;
};

const Translation *getTranslationTable();

class TranslationTable {
public:
  static std::string getTranslation(const char *key) {

    if (map_config.size() == 0)
      TranslationTable::init();

    return map_config[key];
  }

private:
  static void init() {

    const Translation *translations = getTranslationTable();

    int i = 0;
    while (strcmp(translations[i].key, "TRANSLATION_TABLE_END")) {
      std::cout << "Mapping " << translations[i].value << " to "
                << translations[i].key << std::endl;
      map_config[translations[i].key] = translations[i].value;
      ++i;
    }
  }

  static std::map<std::string, std::string> map_config;
};

std::map<std::string, std::string> TranslationTable::map_config;

#endif
