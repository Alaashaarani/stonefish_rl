from receiver import Receiver
from time import sleep

print("Starting Receiver examples...")
# Example 1: Continuous receiving with printing
receiver = Receiver()
# receiver.start()  # Blocks and prints all messages
print("after receiver started")

# Example 2: Disable printing, use return_msg()
receiver = Receiver()
receiver.enable_printing_msgs(False)  # Turn off printing

while True:
    msg = receiver.return_msg()
    if msg:
        print(f"Received: {msg['title']} = {msg['data']} (type: {msg['type']}) (ID: {msg['id']})") 
        print("*********************Processing message... in layers***********************")

    # Do other work...

# Example 3: Mixed usage
receiver = Receiver()
receiver.enable_printing_msgs(True)  # Enable printing

# Get one message without blocking forever
msg = receiver.return_msg()
if msg:
    print(f"First message: {msg}")

# Then start continuous receiving
receiver.start()