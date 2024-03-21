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

#include <memory>
#include <mutex>
#include <deque>
#include <functional>
#include <map>
#include <thread>
#include <sstream>

#include "loadgen.h"

#ifdef NETWORK_DIVISION
#include "query_dispatch_library.h"
typedef mlperf::QueryDispatchLibrary TestBase;
std::string QDLName = "KILT_QDL";
#else // everything else
#include "query_sample_library.h"
typedef mlperf::SystemUnderTest TestBase;
#endif

#include "system_under_test.h"
#include "test_settings.h"

#include "config/harness_config.h"
#include "kilt.h"

using namespace std;
using namespace KRAI;

std::vector<int> MAX_INPUT_LENGTHS;

class MultiKILT {
public:
  MultiKILT(std::vector<std::string> config_files)
      : _config_files(config_files.begin(), config_files.end()) {}

  void LoadKILT() {
    if (_config_files.size() == 0) {
      return;
    }

    std::scoped_lock lock(_kilt_lock);

    std::string fname = _config_files.front();
    _config_files.pop_front();
    setJSONConfig(fname.c_str());

    _kilts.push_back(std::make_unique<KILT>());
  }

  void UnloadKILT() {
    std::scoped_lock lock(_kilt_lock);
    _kilts.pop_front();
  }

  void Iterate(const std::function<void(KILT *)> &func) {
    std::scoped_lock lock(_kilt_lock);
    for (auto &kilt : _kilts) {
      func(kilt.get());
    }
  }

  KILT *GetKILT(size_t idx) {
    std::scoped_lock lock(_kilt_lock);
    return _kilts.at(idx).get();
  }

  KILT *GetKILT() { return GetKILT(0); }

private:
  std::mutex _kilt_lock;
  std::deque<std::unique_ptr<KILT>> _kilts;
  std::deque<std::string> _config_files;
};

class Testable : public TestBase {
public:
  Testable(std::shared_ptr<MultiKILT> multi_kilt, HarnessConfig *cfg)
      : TestBase(), _multi_kilt(multi_kilt) {
    _cfg = cfg;
    query_counter = 0;
  };

  ~Testable() {}

#ifdef NETWORK_DIVISION
  virtual const std::string &LocalName() { return QDLName; }
#endif

  const std::string &Name() override {
    return _multi_kilt->GetKILT()->UniqueServerID();
  }

  void IssueQuery(const std::vector<mlperf::QuerySample> &samples) {

    ++query_counter;
    auto vl = _cfg->getVerbosity();
    if (vl > 1) {
      cout << query_counter << ") IssueQuery([" << samples.size() << "],"
           << samples[0].id << "," << samples[0].index << ")" << endl;
    } else if (vl) {
      cout << 'Q' << flush;
    }

    std::map<int, std::vector<mlperf::QuerySample>> bins;

    for (auto max_inp_len : MAX_INPUT_LENGTHS) {
      bins.emplace(max_inp_len, std::vector<mlperf::QuerySample>());
    }

    // Create a temporary data source for binning
    IConfig config;
    auto ds = dataSourceConstruct(&config,
                                  config.server_cfg->getDataSourceAffinity(0));
    ds->loadSamples(nullptr);

    for (auto s : samples) {
      int32_t input_len = *static_cast<int32_t *>(ds->getSamplePtr(s.index, 2));

      for (auto max_input_length : MAX_INPUT_LENGTHS) {
        if (input_len < max_input_length) {
          bins[max_input_length].push_back(s);
          break;
        }
      }
    }

    std::vector<std::thread> threads;

    for (int i = 0; i < bins.size(); i++) {
      threads.emplace_back(
          [&](size_t kilt_idx, std::vector<mlperf::QuerySample> bin) {
            // Sort the bin
            std::sort(bin.begin(), bin.end(),
                      [&ds](mlperf::QuerySample a, mlperf::QuerySample b) {
                        int32_t input_len_a = *static_cast<int32_t *>(
                            ds->getSamplePtr(a.index, 2));
                        int32_t input_len_b = *static_cast<int32_t *>(
                            ds->getSamplePtr(b.index, 2));
                        return input_len_a < input_len_b;
                      });

            auto kilt = _multi_kilt->GetKILT(kilt_idx);

            std::cout << "BIN: ";
            for (auto s : bin) {
              std::cout << s.index << " ";
            }
            std::cout << std::endl;

            kilt->LoadNextBatch(&bin);
            kilt->Inference(bin);
            // int completed = 0;
            // while (completed < bin.size()) {
            //   completed = kilt->GetCompletedSampleCount();
            //   std::this_thread::sleep_for(std::chrono::milliseconds(100));
            // }
          },
          i, bins[MAX_INPUT_LENGTHS[i]]);

      // _multi_kilt->UnloadKILT();
      // _multi_kilt->LoadKILT();
    }

    for (auto &t : threads) {
      t.join();
    }
  }

  void FlushQueries() {
    auto vl = _cfg->getVerbosity();
    if (vl) {
      cout << endl;
    }
  }

private:
  std::string _name{"QAIC_SUT"};
  std::shared_ptr<MultiKILT> _multi_kilt;
  HarnessConfig *_cfg;
  long query_counter;
  mlperf::TestScenario scenario;
};

class QuerySampleLibraryQAIC : public mlperf::QuerySampleLibrary {
public:
  QuerySampleLibraryQAIC(std::shared_ptr<MultiKILT> multi_kilt,
                         HarnessConfig *cfg)
      : mlperf::QuerySampleLibrary(), _multi_kilt(multi_kilt) {
    _cfg = cfg;
  };

  ~QuerySampleLibraryQAIC() = default;

  const std::string &Name() override { return _name; }

  size_t TotalSampleCount() override {
    return _multi_kilt->GetKILT()->AvailableSamplesMax();
  }

  size_t PerformanceSampleCount() override {
    return _multi_kilt->GetKILT()->SamplesInMemoryMax();
  }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override {
    _multi_kilt->Iterate([&samples](KILT *kilt) {
      kilt->LoadNextBatch(
          const_cast<std::vector<mlperf::QuerySampleIndex> *>(&samples));
    });
  }

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override {
    _multi_kilt->Iterate([&samples](KILT *kilt) {
      kilt->UnloadBatch(
          const_cast<std::vector<mlperf::QuerySampleIndex> *>(&samples));
    });
  }

private:
  std::string _name{"QAIC_QSL"};
  std::shared_ptr<MultiKILT> _multi_kilt;
  HarnessConfig *_cfg;
};

void Test(std::shared_ptr<MultiKILT> multi_kilt, HarnessConfig *cfg) {

  const std::string mlperf_conf_path = cfg->getMLPerfConfigPath();
  const std::string user_conf_path = cfg->getUserConfPath();
  const std::string model_name = cfg->getModelName();

  const std::string scenario_string = cfg->getLoadgenScenario();
  const std::string mode_string = cfg->getLoadgenMode();

  std::cout << "Path to mlperf.conf : " << mlperf_conf_path << std::endl;
  std::cout << "Path to user.conf : " << user_conf_path << std::endl;
  std::cout << "Model Name: " << model_name << std::endl;
  std::cout << "LoadGen Scenario: " << scenario_string << std::endl;
  std::cout << "LoadGen Mode: "
            << (mode_string != "" ? mode_string : "(empty string)")
            << std::endl;

  mlperf::TestSettings ts;

  // This should have been done automatically inside ts.FromConfig() !
  ts.scenario =
      (scenario_string == "SingleStream")  ? mlperf::TestScenario::SingleStream
      : (scenario_string == "MultiStream") ? mlperf::TestScenario::MultiStream
      : (scenario_string == "Server")      ? mlperf::TestScenario::Server
      : (scenario_string == "Offline")     ? mlperf::TestScenario::Offline
                                           : mlperf::TestScenario::SingleStream;

  if (mode_string != "")
    ts.mode = (mode_string == "SubmissionRun") ? mlperf::TestMode::SubmissionRun
              : (mode_string == "AccuracyOnly") ? mlperf::TestMode::AccuracyOnly
              : (mode_string == "PerformanceOnly")
                  ? mlperf::TestMode::PerformanceOnly
              : (mode_string == "FindPeakPerformance")
                  ? mlperf::TestMode::FindPeakPerformance
                  : mlperf::TestMode::SubmissionRun;

  if (ts.FromConfig(mlperf_conf_path, model_name, scenario_string)) {
    std::cout << "Issue with mlperf.conf file at " << mlperf_conf_path
              << std::endl;
    exit(1);
  }

  if (ts.FromConfig(user_conf_path, model_name, scenario_string)) {
    std::cout << "Issue with user.conf file at " << user_conf_path << std::endl;
    exit(1);
  }

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = false;
  log_settings.enable_trace = false;

  if (cfg->triggerColdRun()) {
    multi_kilt->GetKILT()->ColdRun();
  }

  Testable testable(multi_kilt, cfg);
  QuerySampleLibraryQAIC qsl(multi_kilt, cfg);

  mlperf::StartTest(&testable, &qsl, ts, log_settings);
}

std::vector<std::string> splitString(const std::string &inputString) {
  std::istringstream iss(inputString);
  std::vector<std::string> tokens;
  std::string token;

  while (std::getline(iss, token, ':')) {
    tokens.push_back(token);
  }

  return tokens;
}

int main(int argc, char *argv[]) {
  try {
    HarnessConfig *cfg = new HarnessConfig();

    std::string config_file_paths_str(getenv("KILT_CONFIG_PATHS"));
    std::string bin_sizes_str(getenv("KILT_BIN_SIZES"));

    auto config_file_paths = splitString(config_file_paths_str);
    auto bin_sizes = splitString(bin_sizes_str);

    for (auto bin_size : bin_sizes) {
      MAX_INPUT_LENGTHS.push_back(std::stoi(bin_size));
    }

    std::shared_ptr<MultiKILT> multi_kilt =
        std::make_shared<MultiKILT>(config_file_paths);

    for (int i = 0; i < bin_sizes.size(); i++) {
      multi_kilt->LoadKILT();
    }

    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    for (int i = 0; i < 128; ++i)
      CPU_SET(i, &cpu_set);

    pthread_t t = pthread_self();
    pthread_setaffinity_np(t, sizeof(cpu_set_t), &cpu_set);

    Test(multi_kilt, cfg);
    delete cfg;
  } catch (const string &error_message) {
    cerr << "ERROR: " << error_message << endl;
    return -1;
  }
  return 0;
}
