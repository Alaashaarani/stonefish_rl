#include "Sender.h"
#include <iostream>
#include <sstream>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

Sender::Sender(const std::string& address) 
    : context(1), socket(context, zmq::socket_type::rep) {
    
    try{
    socket.bind(address);
    std::cout << "[ZMQ Log] Sender bound to: " << address << std::endl;
    }
    catch(const zmq::error_t &e)
    {
        std::cerr << "[ZMQ ERROR] " << e.what() << "\n";
    }

    #ifdef _WIN32
        Sleep(1000);
    #else
        sleep(1);
    #endif
    
    std::cout << "[ZMQ Log] Sender ready!" << std::endl;
}

Sender::~Sender() {
    socket.close();
    context.close();

    std::cout << "[ZMQ Log] Sender shutting down..." << std::endl;    
}

// Template implementation for simple types
template<typename T>
void Sender::send(const std::string& title, const T& data, int id) {
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
    
    std::cout << "Sent - ID: " << id << ", Title: " << title << ", Value: " << data << std::endl;
}

// Explicit template instantiations
template void Sender::send<float>(const std::string&, const float&, int);
template void Sender::send<int>(const std::string&, const int&, int);
template void Sender::send<double>(const std::string&, const double&, int);
template void Sender::send<bool>(const std::string&, const bool&, int);

// Specialization for std::vector<float>
void Sender::send(const std::string& title, const std::vector<float>& data, int id) {
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    zmq::message_t data_msg(data.size() * sizeof(float));
    memcpy(data_msg.data(), data.data(), data.size() * sizeof(float));
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "Sent - ID: " << id << ", Title: " << title << ", vector<float> size: " << data.size() << std::endl;
}

// Specialization for std::vector<int>
void Sender::send(const std::string& title, const std::vector<int>& data, int id) {
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    zmq::message_t data_msg(data.size() * sizeof(int));
    memcpy(data_msg.data(), data.data(), data.size() * sizeof(int));
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "Sent - ID: " << id << ", Title: " << title << ", vector<int> size: " << data.size() << std::endl;
}

// Specialization for std::vector<double>
void Sender::send(const std::string& title, const std::vector<double>& data, int id) {
    zmq::message_t id_msg(sizeof(id));
    memcpy(id_msg.data(), &id, sizeof(id));
    
    zmq::message_t title_msg(title.size());
    memcpy(title_msg.data(), title.c_str(), title.size());
    
    zmq::message_t data_msg(data.size() * sizeof(double));
    memcpy(data_msg.data(), data.data(), data.size() * sizeof(double));
    
    socket.send(id_msg, zmq::send_flags::sndmore);
    socket.send(title_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
    
    std::cout << "Sent - ID: " << id << ", Title: " << title << ", vector<double> size: " << data.size() << std::endl;
}

// Specialization for std::vector<std::string>
void Sender::send(const std::string& title, const std::vector<std::string>& data, int id) {
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
    
    std::cout << "Sent - ID: " << id << ", Title: " << title << ", vector<string> size: " << data.size() << std::endl;
}

zmq::message_t Sender::receive() {
    zmq::message_t msg;
    auto recv_result = socket.recv(msg);
    std::cout << "[ZMQ Log] Received message of size: " << msg.size() << "and return" << " bytes." << std::endl;

    return msg;
}

bool Sender::receive(zmq::message_t& msg, zmq::recv_flags flags) {
    auto result = socket.recv(msg, flags);
    return result.has_value();
}

void Sender::sendJson(const std::string& json_str) {
    this->socket.send(zmq::buffer(json_str), zmq::send_flags::none);
}