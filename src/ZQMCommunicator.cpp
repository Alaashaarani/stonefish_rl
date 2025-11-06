#include "ZMQCommunicator.h"
#include <iostream>
#include <sstream>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

ZMQCommunicator::ZMQCommunicator(const std::string& address) 
    : context(1), socket(context, ZMQ_REP) {
    
    socket.bind(address);
    std::cout << "[ZMQ] Communicator bound to: " << address << std::endl;
    
    #ifdef _WIN32
        Sleep(1000);
    #else
        sleep(1);
    #endif
    
    std::cout << "[ZMQ] Communicator ready!" << std::endl;
}

// Template implementation for simple types
template<typename T>
void ZMQCommunicator::send(const std::string& title, const T& data, int id) {
    // Message parts: [ID, Title, Data]
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    zmq::message_t data_msg(sizeof(data));
    memcpy(data_msg.data(), &data, sizeof(data));
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "[ZMQ] Sent - ID: " << id << ", Title: " << title << std::endl;
}

// Explicit template instantiations
template void ZMQCommunicator::send<float>(const std::string&, const float&, int);
template void ZMQCommunicator::send<int>(const std::string&, const int&, int);
template void ZMQCommunicator::send<double>(const std::string&, const double&, int);
template void ZMQCommunicator::send<bool>(const std::string&, const bool&, int);

// Specialization for std::vector<float>
void ZMQCommunicator::send(const std::string& title, const std::vector<float>& data, int id) {
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    zmq::message_t data_msg(data.size() * sizeof(float));
    memcpy(data_msg.data(), data.data(), data.size() * sizeof(float));
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "[ZMQ] Sent vector<float> - ID: " << id << ", Size: " << data.size() << std::endl;
}

// Specialization for std::vector<int>
void ZMQCommunicator::send(const std::string& title, const std::vector<int>& data, int id) {
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    zmq::message_t data_msg(data.size() * sizeof(int));
    memcpy(data_msg.data(), data.data(), data.size() * sizeof(int));
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "[ZMQ] Sent vector<int> - ID: " << id << ", Size: " << data.size() << std::endl;
}

// Specialization for std::vector<double>
void ZMQCommunicator::send(const std::string& title, const std::vector<double>& data, int id) {
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    zmq::message_t data_msg(data.size() * sizeof(double));
    memcpy(data_msg.data(), data.data(), data.size() * sizeof(double));
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "[ZMQ] Sent vector<double> - ID: " << id << ", Size: " << data.size() << std::endl;
}

// Specialization for std::vector<std::string>
void ZMQCommunicator::send(const std::string& title, const std::vector<std::string>& data, int id) {
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    // For string vectors, serialize them
    std::stringstream ss;
    for (size_t i = 0; i < data.size(); ++i) {
        ss << data[i];
        if (i != data.size() - 1) {
            ss << "|";  // Use pipe as delimiter
        }
    }
    
    std::string serialized = ss.str();
    zmq::message_t data_msg(serialized.size());
    memcpy(data_msg.data(), serialized.c_str(), serialized.size());
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "[ZMQ] Sent vector<string> - ID: " << id << ", Size: " << data.size() << std::endl;
}

// Send JSON string
void ZMQCommunicator::sendJson(const std::string& json_str) {
    zmq::message_t msg(json_str.size());
    memcpy(msg.data(), json_str.c_str(), json_str.size());
    socket.send(msg, zmq::send_flags::none);
    std::cout << "[ZMQ] Sent JSON: " << json_str.length() << " bytes" << std::endl;
}

// Receive message
zmq::message_t ZMQCommunicator::receive() {
    zmq::message_t msg;
    
    try {
        auto result = socket.recv(msg, zmq::recv_flags::none);
        
        if (!result) {
            std::cerr << "[ZMQ] Receive failed - no message received" << std::endl;
            return zmq::message_t(0);
        }
        
        std::cout << "[ZMQ] Received: " << msg.size() << " bytes" << std::endl;
        return msg;
        
    } catch (const zmq::error_t& e) {
        std::cerr << "[ZMQ] Error receiving message: " << e.what() << std::endl;
        return zmq::message_t(0);
    }
}

// Receive with flags
bool ZMQCommunicator::receive(zmq::message_t& msg, zmq::recv_flags flags) {
    try {
        auto result = socket.recv(msg, flags);
        
        if (result) {
            std::cout << "[ZMQ] Received: " << msg.size() << " bytes" << std::endl;
            return true;
        } else {
            // This is normal for non-blocking receives with no message
            if (flags == zmq::recv_flags::dontwait) {
                // Don't print warning for non-blocking with no message
            } else {
                std::cout << "[ZMQ] Receive failed" << std::endl;
            }
            return false;
        }
        
    } catch (const zmq::error_t& e) {
        std::cerr << "[ZMQ] Error receiving message: " << e.what() << std::endl;
        return false;
    }
}

ZMQCommunicator::~ZMQCommunicator() {
    std::cout << "[ZMQ] Communicator shutting down..." << std::endl;
}