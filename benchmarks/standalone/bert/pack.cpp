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
#include <iostream>
#include <map>
#include <vector>

#include "pack.h"

void add_pack(std::vector<int> &pack, int count, Lookup &tmp, Lookup &complete,
              int limit, int offset) {

  if (pack.size() == limit || offset == 0) {
    complete[offset].push_back(Entry(count, pack));
  } else {
    tmp[offset].push_back(Entry(count, pack));
  }
}

void histify(const std::vector<SizedSample> &samples,
             std::vector<std::vector<SizedSample>> &histogram) {

  for (int i = 0; i < samples.size(); ++i) {
    histogram[samples[i].second - 1].push_back(samples[i]);
  }
}

void pack_samples(Lookup &lookup,
                  std::vector<std::vector<SizedSample>> &histogram,
                  std::vector<std::vector<SizedSample>> &packed_samples) {

  for (auto lit = lookup.begin(); lit != lookup.end(); lit++) {
    for (auto git = lit->second.begin(); git != lit->second.end(); git++) {
      for (int c = 0; c < git->first; ++c) {
        std::vector<SizedSample> vqs;
        for (auto bit = git->second.begin(); bit != git->second.end(); bit++) {
          vqs.push_back(histogram[(*bit) - 1].back());
          histogram[(*bit) - 1].pop_back();
        }
        packed_samples.push_back(vqs);
      }
    }
  }

  int count = 0;
  for (auto xx = packed_samples.begin(); xx != packed_samples.end(); xx++) {
    count += xx->size();
  }

  count = 0;
  for (auto xx = histogram.begin(); xx != histogram.end(); xx++) {
    count += xx->size();
  }
}

void pack(const std::vector<SizedSample> &samples, int max_seq_len,
          int max_seq_per_pack,
          std::vector<std::vector<SizedSample>> &packed_samples) {

  std::vector<std::vector<SizedSample>> histogram(max_seq_len);

  histify(samples, histogram);

  Lookup strategies_per_length;
  Lookup tmp_strategies_per_length;

  for (int i = 0; i < max_seq_len; ++i) {
    int n_sequences_to_bin = histogram[(max_seq_len - 1) - i].size();
    int length_to_bin = max_seq_len - i;
    int offset = i + 1;
    while (n_sequences_to_bin > 0) {
      auto val = tmp_strategies_per_length.find(length_to_bin + offset);
      if (val != tmp_strategies_per_length.end()) {
        auto [n_sequences_to_pack, pack] = val->second.back();
        val->second.pop_back();
        auto new_pack = pack;
        new_pack.push_back(length_to_bin);
        auto count = std::min(n_sequences_to_pack, n_sequences_to_bin);
        if (n_sequences_to_pack > n_sequences_to_bin) {
          n_sequences_to_pack -= n_sequences_to_bin;
          tmp_strategies_per_length[length_to_bin + offset].push_back(
              Entry(n_sequences_to_pack, pack));
          n_sequences_to_bin = 0;
        } else {
          n_sequences_to_bin -= n_sequences_to_pack;
        }
        add_pack(new_pack, count, tmp_strategies_per_length,
                 strategies_per_length, max_seq_per_pack, offset);
        if (tmp_strategies_per_length[length_to_bin + offset].empty())
          tmp_strategies_per_length.erase(length_to_bin + offset);
      } else {
        offset -= 1;
      }

      if (offset < 0) {
        std::vector<int> vec({length_to_bin});
        add_pack(vec, n_sequences_to_bin, tmp_strategies_per_length,
                 strategies_per_length, max_seq_per_pack, i);
        n_sequences_to_bin = 0;
      }
    }
  }

  for (auto lit = tmp_strategies_per_length.begin();
       lit != tmp_strategies_per_length.end(); lit++) {
    strategies_per_length[lit->first].insert(
        strategies_per_length[lit->first].end(),
        tmp_strategies_per_length[lit->first].begin(),
        tmp_strategies_per_length[lit->first].end());
  }

  pack_samples(strategies_per_length, histogram, packed_samples);

  // std::cout << "Pack Ratio: " << samples.size() << " " <<
  // packed_samples.size() << " " << (float)samples.size()/packed_samples.size()
  // << std::endl;
}
