#include "ZMQCommunicator.h"
#include <iostream>
#include <sstream>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

// Gemini Constructor 
ZMQCommunicator::ZMQCommunicator(int port)
    : context(1), socket(context, ZMQ_REP), m_port(port) { // Initialize m_port here

    std::string address = "tcp://*:" + std::to_string(m_port);
    try {
        // Use the universal setsockopt syntax here too
        int linger = 0;
        socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));

        socket.bind(address);
        std::cout << "[ZMQCommunicator] Listening on " << address << std::endl;
        // ... rest of your code ...
    } catch (const zmq::error_t& e) {
        std::cerr << "[ZMQ ERROR] Could not bind: " << e.what() << std::endl;
    }
}

// Old Constructor
// ZMQCommunicator::ZMQCommunicator(int port)
//     : context(1), socket(context, ZMQ_REP) {

//     // Bind to specified port
//     std::string address = "tcp://*:" + std::to_string(port);
//     try {
//         socket.bind(address);
//         std::cout << "[ZMQCommunicator] Listening on " << address << std::endl;
        
//         #ifdef _WIN32
//             Sleep(500);
//         #else
//             usleep(500000); // 0.5 seconds
//         #endif
        
//     } catch (const zmq::error_t& e) {
//         std::cerr << "[ZMQ ERROR] Could not bind to " << address << ": " << e.what() << std::endl;
//     }
// }

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
    // debug output
    // std::cout << "[ZMQ] Sent JSON: " << json_str.length() << " bytes" << std::endl;
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
        // debug output
        // std::cout << "[ZMQ] Received: " << msg.size() << " bytes" << std::endl;
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
            // debug output
            // std::cout << "[ZMQ] Received: " << msg.size() << " bytes" << std::endl;
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
        std::cerr << "[ZMQ] Error receiving message with flag: " << e.what() << std::endl;
        return false;
    }
}

void ZMQCommunicator::reset() {
    std::cout << "[ZMQ] Resetting socket and clearing buffers..." << std::endl;

    // 1. Force immediate drop of pending messages
    int linger = 0;
    socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    
    // 2. Close the socket
    socket.close();

    // 3. MANDATORY: Wait for the OS to release the port
    // 100ms is usually enough to clear the 'Address already in use' conflict
    #ifdef _WIN32
        Sleep(100);
    #else
        usleep(100000); // 100,000 microseconds = 0.1 seconds
    #endif

    // 4. Re-create the socket on the existing context
    socket = zmq::socket_t(context, ZMQ_REP);

    // 5. Re-apply safety settings to the NEW socket
    socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    int hwm = 100; 
    socket.setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
    socket.setsockopt(ZMQ_RCVHWM, &hwm, sizeof(hwm));

    // 6. Re-bind
    std::string address = "tcp://*:" + std::to_string(m_port);
    try {
        socket.bind(address);
        std::cout << "[ZMQ] Socket successfully restarted on port " << m_port << std::endl;
    } catch (const zmq::error_t& e) {
        // If it still fails, we log it, but the Sleep usually fixes this
        std::cerr << "[ZMQ ERROR] Reset bind failed: " << e.what() << std::endl;
    }
}


ZMQCommunicator::~ZMQCommunicator() {
    std::cout << "[ZMQ] Communicator shutting down..." << std::endl;
    try {
        socket.close();
        context.close();
    } catch (const zmq::error_t& e) {
        // Log it if needed
    }
}