import ctypes
import time


WM_KEYDOWN = 0x100
WM_KEYUP = 0x101


def send_key(key, hold: int = None):
    # Constants

    VK_KEYS = {
        "W": 0x57,
        "A": 0x51,
        "S": 0x53,
        "D": 0x45,
        "1": 0x31,
        "Q" : 0x41,
        "E" : 0x44,
    }

    # Find the window
    hwnd = ctypes.windll.user32.FindWindowW(None, "World of Warcraft")

    if hwnd:
        # Bring the window to the foreground
        ctypes.windll.user32.SetForegroundWindow(hwnd)

        # Send a keydown event for the specified key
        ctypes.windll.user32.PostMessageW(hwnd, WM_KEYDOWN, VK_KEYS[key.upper()], 0)

        if hold:
            time.sleep(hold)

        # Continuously check the condition to hold the key
        # Send a keyup event for the specified key
        ctypes.windll.user32.PostMessageW(hwnd, WM_KEYUP, VK_KEYS[key.upper()], 0)
    else:
        print("Window not found!")
