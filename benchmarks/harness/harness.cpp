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

class Testable : public TestBase {
public:
  Testable(KILT *kil, HarnessConfig *cfg) : TestBase() {
    _kil = kil;
    _cfg = cfg;
    query_counter = 0;
  };

  ~Testable() {}

#ifdef NETWORK_DIVISION
  virtual const std::string &LocalName() { return QDLName; }
#endif

  const std::string &Name() override { return _kil->UniqueServerID(); }

  void IssueQuery(const std::vector<mlperf::QuerySample> &samples) {

    ++query_counter;
    auto vl = _cfg->getVerbosity();
    if (vl > 1) {
      cout << query_counter << ") IssueQuery([" << samples.size() << "],"
           << samples[0].id << "," << samples[0].index << ")" << endl;
    } else if (vl) {
      cout << 'Q' << flush;
    }

    _kil->Inference(samples);
  }

  void FlushQueries() {
    auto vl = _cfg->getVerbosity();
    if (vl) {
      cout << endl;
    }
  }

private:
  std::string _name{"QAIC_SUT"};
  KILT *_kil;
  HarnessConfig *_cfg;
  long query_counter;
  mlperf::TestScenario scenario;
};

class QuerySampleLibraryQAIC : public mlperf::QuerySampleLibrary {
public:
  QuerySampleLibraryQAIC(KILT *kil, HarnessConfig *cfg)
      : mlperf::QuerySampleLibrary() {
    _kil = kil;
    _cfg = cfg;
  };

  ~QuerySampleLibraryQAIC() = default;

  const std::string &Name() override { return _name; }

  size_t TotalSampleCount() override { return _kil->AvailableSamplesMax(); }

  size_t PerformanceSampleCount() override {
    return _kil->SamplesInMemoryMax();
  }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override {
    _kil->LoadNextBatch(
        const_cast<std::vector<mlperf::QuerySampleIndex> *>(&samples));
    return;
  }

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override {
    _kil->UnloadBatch(
        const_cast<std::vector<mlperf::QuerySampleIndex> *>(&samples));
    return;
  }

private:
  std::string _name{"QAIC_QSL"};
  KILT *_kil;
  HarnessConfig *_cfg;
};

void Test(KILT *kil, HarnessConfig *cfg) {

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
    kil->ColdRun();
  }

  Testable testable(kil, cfg);
  QuerySampleLibraryQAIC qsl(kil, cfg);

  mlperf::StartTest(&testable, &qsl, ts, log_settings);
}

int main(int argc, char *argv[]) {
  try {
    HarnessConfig *cfg = new HarnessConfig();
    KILT *kil = new KILT;

    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    for (int i = 0; i < 128; ++i)
      CPU_SET(i, &cpu_set);

    pthread_t t = pthread_self();
    pthread_setaffinity_np(t, sizeof(cpu_set_t), &cpu_set);

    Test(kil, cfg);
    delete kil;
    delete cfg;
  } catch (const string &error_message) {
    cerr << "ERROR: " << error_message << endl;
    return -1;
  }
  return 0;
}
