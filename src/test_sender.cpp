#include "Sender.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

int main() {
    Sender sender("tcp://*:5555");
    
    int msg_id = 1;
    
    while (true) {
        // Simple types with titles
        sender.send("temperature", 23.5f, msg_id++);
        sender.send("counter", 42, msg_id++);
        sender.send("pi_value", 3.14159, msg_id++);
        sender.send("is_ready", true, msg_id++);
        
        // Vectors with titles
        std::vector<float> sensor_data = {1.1f, 2.2f, 3.3f, 4.4f};
        sender.send("sensor_readings", sensor_data, msg_id++);
        
        std::vector<int> counters = {10, 20, 30, 40, 50};
        sender.send("device_counters", counters, msg_id++);
        
        std::vector<double> precise_data = {1.5, 2.5, 3.5, 4.5};
        sender.send("precise_measurements", precise_data, msg_id++);
        
        std::vector<std::string> messages = {"hello", "world", "from", "cpp"};
        sender.send("text_messages", messages, msg_id++);
        
        std::cout << "Cycle completed. Last ID: " << msg_id - 1 << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        #ifdef _WIN32
            Sleep(3000);
        #else
            sleep(3);
        #endif
    }
    
    return 0;
}