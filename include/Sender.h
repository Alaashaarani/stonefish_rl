#ifndef SENDER_H
#define SENDER_H

#include <zmq.hpp>
#include <string>
#include <vector>

class Sender {
public:
    Sender(const std::string& address);
    
    // For simple types (int, float, double, bool, etc.)
    template<typename T>
    void send(const std::string& title, const T& data, int id);
    
    // Special handling for vectors
    void send(const std::string& title, const std::vector<float>& data, int id);
    void send(const std::string& title, const std::vector<int>& data, int id);
    void send(const std::string& title, const std::vector<double>& data, int id);
    void send(const std::string& title, const std::vector<std::string>& data, int id);

    // Send Json string
    void sendJson(const std::string& json_str);

     // Add receive capability
    zmq::message_t receive();
    bool receive(zmq::message_t& msg, zmq::recv_flags flags = zmq::recv_flags::none);
    
    ~Sender();

private:
    zmq::context_t context;
    zmq::socket_t socket;
};

#endif // SENDER_H