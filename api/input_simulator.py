import ctypes
import time
import threading

def send_key(key, should_hold):
    # Constants
    WM_KEYDOWN = 0x100
    WM_KEYUP = 0x101
    VK_KEYS = {
        'W': 0x57,
        'A': 0x51,
        'S': 0x53,
        'D': 0x45
    }

    # Find the window
    hwnd = ctypes.windll.user32.FindWindowW(None, "World of Warcraft")

    if hwnd:
        # Bring the window to the foreground
        ctypes.windll.user32.SetForegroundWindow(hwnd)

        # Send a keydown event for the specified key
        ctypes.windll.user32.PostMessageW(hwnd, WM_KEYDOWN, VK_KEYS[key.upper()], 0)

        # Continuously check the condition to hold the key
        while should_hold():
            time.sleep(0.01) # Polling interval

        # Send a keyup event for the specified key
        ctypes.windll.user32.PostMessageW(hwnd, WM_KEYUP, VK_KEYS[key.upper()], 0)
    else:
        print("Window not found!")

def condition():
    # Replace this with your actual condition for holding down the key
    return True

# You can call this function in its own thread
def send_w(condition):
    send_key('W', condition)

def send_a(condition):
    send_key('A', condition)

def send_s(condition):
    send_key('S', condition)

def send_d(condition):
    send_key('D', condition)

def send_one():
    # Constants
    WM_KEYDOWN = 0x100
    WM_KEYUP = 0x101

    # Find the window
    hwnd = ctypes.windll.user32.FindWindowW(None, "World of Warcraft")

    if hwnd:
        # Bring the window to the foreground
        ctypes.windll.user32.SetForegroundWindow(hwnd)

        # Send the keydown event for the digit '1'
        ctypes.windll.user32.PostMessageW(hwnd, WM_KEYDOWN, ord('1'), 0)
        # Send the keyup event for the digit '1'
        ctypes.windll.user32.PostMessageW(hwnd, WM_KEYUP, ord('1'), 0)
    else:
        print("Window not found!")



#Usage example:
'''
# Create threads for each function
thread_w = threading.Thread(target=send_w)
thread_a = threading.Thread(target=send_a)
thread_s = threading.Thread(target=send_s)
thread_d = threading.Thread(target=send_d)

# Start the threads
thread_w.start()
thread_a.start()
thread_s.start()
thread_d.start()

# Optional: Wait for all threads to complete
thread_w.join()
thread_a.join()
thread_s.join()
thread_d.join()

'''