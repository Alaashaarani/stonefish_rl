#ifndef ZMQCOMMUNICATOR_H
#define ZMQCOMMUNICATOR_H

#include <zmq.hpp>
#include <string>
#include <vector>

class ZMQCommunicator {
public:

    // ZMQCommunicator(const std::string& address = "tcp://*:5555");
    ZMQCommunicator(int port = 5555); // Constructor with port number

    // Send methods for different data types
    template<typename T>
    void send(const std::string& title, const T& data, int id);
    
    // Send yaml string (simple wrapper)
    void sendMessage(const std::string& json_str);
    
    // Receive methods
    bool receive(zmq::message_t& msg);
    
    void reset();
    
    ~ZMQCommunicator();
private:
    zmq::context_t context;
    zmq::socket_t socket;
    int m_port;
};

#endif // ZMQCOMMUNICATOR_H