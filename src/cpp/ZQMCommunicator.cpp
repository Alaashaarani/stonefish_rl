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
    : context(1), socket(context, ZMQ_REP), m_port(port)
{
    std::string address = "tcp://*:" + std::to_string(m_port);

    try {
        socket.bind(address);
        std::cout << "[ZMQ] REP socket bound to " << address << std::endl;
    } catch (const zmq::error_t& e) {
        std::cerr << "[ZMQ ERROR] Failed to bind: " << e.what() << std::endl;
        throw;
    }
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


// Send a text payload over ZMQ. The payload may contain YAML text.
void ZMQCommunicator::sendMessage(const std::string& payload) {
    socket.send(zmq::buffer(payload), zmq::send_flags::none);
}


// Receive with flags
bool ZMQCommunicator::receive(zmq::message_t& msg) {
    try {
        auto result = socket.recv(msg, zmq::recv_flags::none);
        return result.has_value();
    }
    catch (const zmq::error_t& e) {
        std::cerr << "[ZMQ] receive error: " << e.what() << std::endl;
        throw;
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