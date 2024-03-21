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

#pragma once

namespace KRAI {

//----------------------------------------------------------------------

class NetworkClientConfig {
public:
  const std::string getNetworkServerIPAddress() { return server_ip_address; }

  const int getNetworkServerPort() { return server_port; }

  const int getVerbosityLevel() { return verbosity_level; }

  const int getNumSockets() { return num_sockets; }

  const int getPayloadSize() { return payload_size; }

private:
  const int num_sockets = getconfig_i("KILT_NETWORK_NUM_SOCKETS");

  const int verbosity_level = getconfig_i("KILT_VERBOSE");

  const std::string localhost = "127.0.0.1";
  const std::string server_ip_address =
      alter_str(getconfig_c("KILT_NETWORK_SERVER_IP_ADDRESS"), localhost);

  const int server_port =
      alter_str_i(getconfig_c("KILT_NETWORK_SERVER_PORT"), 8080);

  const int payload_size = getconfig_i("KILT_NETWORK_PAYLOAD_SIZE");
};

class NetworkServerConfig {
public:
  const int getNetworkServerPort() { return port; }

  const int getVerbosityLevel() { return verbosity_level; }

  const int getNumSockets() { return num_sockets; }

  const int verbosity_level = getconfig_i("KILT_VERBOSE");
  const int port = alter_str_i(getconfig_c("KILT_NETWORK_SERVER_PORT"), 8080);
  const int num_sockets = getconfig_i("KILT_NETWORK_NUM_SOCKETS");
};
} // namespace KRAI
