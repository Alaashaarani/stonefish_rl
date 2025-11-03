import zmq
import struct

class Receiver:
    def __init__(self, address="tcp://localhost:5556"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.running = False
        self.print_msgs = True  # Default to printing messages
        
    def enable_printing_msgs(self, enable=True):
        """Enable or disable printing received messages"""
        self.print_msgs = enable
        print(f"Printing messages set to: {self.print_msgs}")
        
    def return_msg(self):
        """Receive and return one message without blocking forever"""
        try:
            # Set a short timeout to avoid blocking forever
            self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            
            # Receive 3-part message: [ID, Title, Data]
            id_data = self.socket.recv()
            title_data = self.socket.recv(zmq.RCVMORE)
            data_msg = self.socket.recv()
            
            # Process and return the message
            return self._process_message(id_data, title_data, data_msg, return_data=True)
            
        except zmq.Again:
            # No message available
            return None
        except Exception as e:
            if self.print_msgs:
                print(f"Error receiving message: {e}")
            return None
    
    def start(self):
        """Start receiving messages continuously (blocking)"""
        self.running = True
        # Remove timeout for continuous receiving
        self.socket.setsockopt(zmq.RCVTIMEO, -1)  # Block forever
        
        if self.print_msgs:
            print("Receiver started, waiting for messages...")
        
        try:
            while self.running:
                # Receive 3-part message: [ID, Title, Data]
                id_data = self.socket.recv()
                title_data = self.socket.recv(zmq.RCVMORE)
                data_msg = self.socket.recv()
                
                self._process_message(id_data, title_data, data_msg)
                
        except KeyboardInterrupt:
            if self.print_msgs:
                print("\nReceiver interrupted by user")
        except Exception as e:
            if self.print_msgs:
                print(f"Receiver error: {e}")
        finally:
            self.stop()
    
    

    
    def _process_message(self, id_data, title_data, data_msg, return_data=False):
        """Process a received message, optionally return data instead of printing"""
        msg_id = struct.unpack('i', id_data)[0]
        title = title_data.decode('utf-8')
        data_size = len(data_msg)
        
        result = {
            'id': msg_id,
            'title': title,
            'type': 'unknown',
            'data': None
        }
        
        # Detect type by size and content
        if data_size == 4:  # Likely float/int
            try:
                value = struct.unpack('f', data_msg)[0]
                result['type'] = 'float'
                result['data'] = value
                if self.print_msgs and not return_data:
                    print(f"ID: {msg_id:>2} | {title:>20} | float: {value:.6f}")
            except:
                value = struct.unpack('i', data_msg)[0]
                result['type'] = 'int'
                result['data'] = value
                if self.print_msgs and not return_data:
                    print(f"ID: {msg_id:>2} | {title:>20} | int: {value}")
                
        elif data_size == 8:  # Likely double
            value = struct.unpack('d', data_msg)[0]
            result['type'] = 'double'
            result['data'] = value
            if self.print_msgs and not return_data:
                print(f"ID: {msg_id:>2} | {title:>20} | double: {value:.10f}")
            
        elif data_size == 1:  # Likely bool
            value = struct.unpack('?', data_msg)[0]
            result['type'] = 'bool'
            result['data'] = value
            if self.print_msgs and not return_data:
                print(f"ID: {msg_id:>2} | {title:>20} | bool: {value}")
            
        elif data_size % 4 == 0 and data_size > 4:  # Likely vector of floats/ints
            num_elements = data_size // 4
            try:
                values = struct.unpack(f'{num_elements}f', data_msg)
                result['type'] = 'vector_float'
                result['data'] = list(values)
                if self.print_msgs and not return_data:
                    print(f"ID: {msg_id:>2} | {title:>20} | vector<float>[{num_elements}]: {[f'{x:.3f}' for x in values]}")
            except:
                values = struct.unpack(f'{num_elements}i', data_msg)
                result['type'] = 'vector_int'
                result['data'] = list(values)
                if self.print_msgs and not return_data:
                    print(f"ID: {msg_id:>2} | {title:>20} | vector<int>[{num_elements}]: {list(values)}")
                
        elif data_size % 8 == 0 and data_size > 8:  # Likely vector of doubles
            num_elements = data_size // 8
            values = struct.unpack(f'{num_elements}d', data_msg)
            result['type'] = 'vector_double'
            result['data'] = list(values)
            if self.print_msgs and not return_data:
                print(f"ID: {msg_id:>2} | {title:>20} | vector<double>[{num_elements}]: {[f'{x:.3f}' for x in values]}")
            
        else:  # Treat as string (for string vectors)
            try:
                value = data_msg.decode('utf-8')
                if '|' in value:  # String vector
                    elements = value.split('|')
                    result['type'] = 'vector_string'
                    result['data'] = elements
                    if self.print_msgs and not return_data:
                        print(f"ID: {msg_id:>2} | {title:>20} | vector<string>[{len(elements)}]: {elements}")
                else:
                    result['type'] = 'string'
                    result['data'] = value
                    if self.print_msgs and not return_data:
                        print(f"ID: {msg_id:>2} | {title:>20} | string: {value}")
            except:
                result['type'] = 'binary'
                result['data'] = data_msg
                if self.print_msgs and not return_data:
                    print(f"ID: {msg_id:>2} | {title:>20} | unknown binary data: {data_size} bytes")
        
        return result if return_data else None
    
    def stop(self):
        """Stop receiving and clean up"""
        self.running = False
        self.socket.close()
        self.context.term()
        if self.print_msgs:
            print("Receiver stopped and cleaned up")

# if __name__ == "__main__":
#     receiver = Receiver()
#     receiver.enable_printing_msgs(False)
#     try:
#         receiver.start()
#     except KeyboardInterrupt:
#         receiver.stop()