#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>

using namespace std;
using namespace boost::asio::ip;

class udp_server {
public:
  udp_server(boost::asio::io_service &io_service)
    : socket_(io_service, udp::endpoint(udp::v4(), 3290)) {
    start_receive();
  }
private:
  void start_receive() {
    socket_.async_receive_from(
      boost::asio::buffer(recv_buffer_), remote_endpoint_,
      boost::bind(&udp_server::handle_receive, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred));
  }

  void handle_receive(const boost::system::error_code &error, std::size_t len) {
    if (!error || error == boost::asio::error::message_size) {
      std::string message;
      message.append(recv_buffer_.c_array(), len);
      if (message != "") cout << message << endl;

      start_receive();
    }
  }

  udp::socket socket_;
  udp::endpoint remote_endpoint_;
  boost::array<char, 256> recv_buffer_;
};

int main(int argc, char *argv[]) {
  try {
    boost::asio::io_service io_service;
    udp_server server(io_service);
    io_service.run();
  } catch(std::exception &e) {
    std::cerr << e.what() << endl;
  }
}
