import numpy as np
from api.frame_capture import WindowsFrameCapture, FrameCapture
import cv2
import time
from api.input_simulator import send_key
from api.injector import get_xyz_coords
from math import sqrt
import torch


class WowEnvironment:
    def __init__(self):
        self.frame_capture = WindowsFrameCapture("World of Warcraft")
        self.action_mapping = {0: "W", 1: "A", 2: "S", 3: "D"}

        self.state = None
        self.current_position = None
        self.target_position = None
        self.distance_to_target = 0

    def starting_position(self):
        # Return the starting position of the agent in the world
        send_key("1")
        time.sleep(1)  # Wait for the game to load
        send_key("q", hold=1)

        self.state = self.get_state_coords()
        self.current_position = get_xyz_coords()
        self.target_position = self.ending_position()

        self.distance_to_target = self.calculate_distance(
            self.current_position, self.target_position
        )


    def calculate_distance(self, point1, point2):
        x1, y1  = point1
        x2, y2 = point2

        distance = sqrt(
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        )
        return distance

    def ending_position(self):
        # Return the ending position, or the target of the agent in the world
        # .go xyz -9449.06 64.8392 56.3581 0
        return (-9018.18, -40.69)

    def get_state_rgb(self):
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
    
    def get_state_coords(self):
        coords = get_xyz_coords()
        normalized_coords = [coords[0] / -10000, coords[1] / 2000]  # Normalize x and y
        return torch.tensor(normalized_coords).unsqueeze(0)

    def step(self, action_num):
    # Previous state and position
        previous_distance_to_target = self.calculate_distance(
            self.current_position, self.target_position
        )
        previous_position = self.current_position
        
        # Apply action
        action = self.action_mapping[action_num]

        if action == "W":
            send_key("W", hold=0.5)
        elif action == "A":
            send_key("A", hold=0.5)
        elif action == "S":
            send_key("S", hold=0.5)
        elif action == "D":
            send_key("D", hold=0.5)

        # Update current position and get the current distance to the target
        self.current_position = get_xyz_coords()
        current_distance_to_target = self.calculate_distance(
            self.current_position, self.target_position
        )

        # Check if the agent has moved after taking the action
        movement_threshold = 0.5  # Define a threshold for movement; adjust as needed
        movement_distance = self.calculate_distance(previous_position, self.current_position)

        # Update reward system
        distance_difference = previous_distance_to_target - current_distance_to_target
        if movement_distance < movement_threshold:
            reward = -50  # Apply a penalty if the agent hasn't moved beyond the threshold
            print("Stuck penalty applied!")
        else:
            # Assign reward based on difference in distance to target. 
            # Closer to target = positive reward, further = negative reward
            reward = distance_difference * 10  # If the agent moved closer, this will be positive

        # Completion Bonus
        done = False
        if current_distance_to_target < 3:
            reward += 1000  # Large reward for reaching the target
            done = True

        # Update state
        self.state = self.get_state_coords()

        return self.state, reward, done


    def reset(self):
        # Reset the world to its initial state
        self.starting_position()
        self.distance = self.calculate_distance(
            self.current_position, self.target_position
        )
     
   
