
from api.frame_capture import WindowsFrameCapture, frame_display


def main():
    frame_capture = WindowsFrameCapture("World of Warcraft")
    frame_display(frame_capture)


if __name__ == "__main__":
    main()