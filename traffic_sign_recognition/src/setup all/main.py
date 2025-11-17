from system import TrafficSignSystem

if __name__ == "__main__":
    IMAGE_PATH = 'C:\\DoAnXuLyAnh\\traffic_sign_recognition\\data\\gstrb-dataset\\gtsrb\\0\\00001_00029.ppm'
    system = TrafficSignSystem()
    system.run(IMAGE_PATH)
