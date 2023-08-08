from abc import ABC, abstractmethod
import win32gui
import win32ui
import win32con
import numpy as np
import cv2
import time

class FrameCapture(ABC):
    @abstractmethod
    def capture(self):
        pass


class WindowsFrameCapture(FrameCapture):
    def __init__(self, title):
        self.hwnd = win32gui.FindWindow(None, title)
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        self.width = right - left
        self.height = bot - top

        self.hwindc = win32gui.GetWindowDC(self.hwnd)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = win32ui.CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)

    def capture(self):
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (self.width, self.height), self.srcdc, (0, 0), win32con.SRCCOPY)

        bmp_info = self.bmp.GetInfo()
        bmp_str = self.bmp.GetBitmapBits(True)
        img_array = np.frombuffer(bmp_str, dtype=np.uint8)
        img_array.shape = (bmp_info['bmHeight'], bmp_info['bmWidth'], 4)

        return img_array[:, :, :3]


def frame_display(frame_capture: FrameCapture):
    while True: # Infinite loop to continuously capture frames
        start_time = time.time() # Record the start time of the frame capture

        frame = frame_capture.capture()
        cv2.imshow("frame", frame)

        end_time = time.time() # Record the end time of the frame capture
        fps = 1 / (end_time - start_time) # Calculate the FPS
        print("FPS:", fps) # Print the FPS to the console

        if cv2.waitKey(1) & 0xFF == ord("q"): # Wait for the "q" key to be pressed
            break

    cv2.destroyAllWindows() # Close the OpenCV window
    return True
