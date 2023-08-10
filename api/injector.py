import socket
import struct


def get_xyz_coords():
    host = "127.0.0.1"
    port = 12345
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(b"Hello, server!")

    data = s.recv(12)  # 3 floats * 4 bytes each

    values = struct.unpack("3f", data)  # 3f because we are expecting 3 floats
    print("Received:", values)
    s.close()
    return values
