#ifndef ZMQCOMMUNICATOR_H
#define ZMQCOMMUNICATOR_H

#include <zmq.hpp>
#include <string>
#include <vector>

class ZMQCommunicator {
public:
    ZMQCommunicator(const std::string& address = "tcp://*:5555");
    
    // Send methods for different data types
    template<typename T>
    void send(const std::string& title, const T& data, int id);
    
    // Special handling for vectors
    void send(const std::string& title, const std::vector<float>& data, int id);
    void send(const std::string& title, const std::vector<int>& data, int id);
    void send(const std::string& title, const std::vector<double>& data, int id);
    void send(const std::string& title, const std::vector<std::string>& data, int id);

    // Send JSON string (simple wrapper)
    void sendJson(const std::string& json_str);

    // Receive methods
    zmq::message_t receive();
    bool receive(zmq::message_t& msg, zmq::recv_flags flags = zmq::recv_flags::none);
    
    ~ZMQCommunicator();

private:
    zmq::context_t context;
    zmq::socket_t socket;
};

#endif // ZMQCOMMUNICATOR_H