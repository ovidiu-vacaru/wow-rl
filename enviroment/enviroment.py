import numpy as np
from api.frame_capture import WindowsFrameCapture, FrameCapture
import cv2
import time
from api.input_simulator import send_w, send_a, send_d, send_s, send_one
import threading
from api.injector import get_xyz_coords
import torch
from math import sqrt


class WowEnvironment:
    def __init__(self, frame_capture: FrameCapture):
        self.action_space = ["W", "A", "S", "D"]
        self.current_position = self.starting_position
        self.target_position = self.ending_position()
        self.frame_capture = frame_capture
        self.current_action = None
        self.state = None
        self.action_mapping = {0: "W", 1: "A", 2: "S", 3: "D"}
        self.distance = 0

    def starting_position(self):
        # Return the starting position of the agent in the world
        send_one()
        time.sleep(2)
        self.state = self.get_state()
        return self.state

    def calculate_distance(self, point1, point2):
        x1, y1, z1 = point1
        x2, y2, z2 = point2

        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance

    def ending_position(self):
        # Return the ending position, or the target of the agent in the world
        # .go xyz -9449.06 64.8392 56.3581 0
        return (-9449.00, 65.00, 56.00)

    def get_state(self):
        # Capture or load the current image state of the environment
        img_array = self.frame_capture.capture()

        # Resize the image to 224x224
        img_array = cv2.resize(img_array, (224, 224))

        # Convert to float32 and normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Transpose the array to the (channels, height, width) shape
        img_array = np.transpose(img_array, (2, 0, 1))

        # Convert the NumPy array to a PyTorch tensor and add a batch dimension
        img_tensor = torch.tensor(img_array).unsqueeze(0)

        return img_tensor

    def step(self, action_num):
        action = self.action_mapping[action_num]
        time.sleep(0.5)  # hard cap waiting time
        self.current_action = action
        if action == "W":
            thread = threading.Thread(
                target=send_w, args=(lambda: self.current_action == "W",)
            )
            thread.start()
        elif action == "A":
            thread = threading.Thread(
                target=send_a, args=(lambda: self.current_action == "A",)
            )
            thread.start()
        elif action == "S":
            thread = threading.Thread(
                target=send_s, args=(lambda: self.current_action == "S",)
            )
            thread.start()
        elif action == "D":
            thread = threading.Thread(
                target=send_d, args=(lambda: self.current_action == "D",)
            )
            thread.start()

        self.current_position = get_xyz_coords()
        self.distance = self.calculate_distance(
            self.current_position, self.target_position
        )
        reward = -1  # Penalize each step to encourage reaching the goal quickly
        done = False
        if self.current_position == self.target_position:
            reward = 10000  # Large reward for reaching the target
            done = True

        self.state = self.get_state()

        return self.state, reward, done

    def reset(self):
        # Reset the world to its initial state
        self.current_position = self.starting_position()
        time.sleep(2)
        self.state = self.get_state()
        return self.state

    def render(self):
        # Optionally, implement a method to render the world visually
        while True:  # Infinite loop to continuously capture frames
            start_time = time.time()  # Record the start time of the frame capture

            frame = self.get_state()
            cv2.imshow("frame", frame)

            end_time = time.time()  # Record the end time of the frame capture
            fps = 1 / (end_time - start_time)  # Calculate the FPS
            print("FPS:", fps)  # Print the FPS to the console

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Wait for the "q" key to be pressed
                break

        cv2.destroyAllWindows()  # Close the OpenCV window
        return True
