import torch

def check_gpu():

    # 检查GPU是否可用
    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available, running on CPU.")
        return 'cpu'

    # 获取当前GPU设备的索引
    device = torch.cuda.current_device()
    print(f"Current device index: {device}")

    # 获取当前GPU设备的名字
    device_name = torch.cuda.get_device_name(device)
    print(f"Device name: {device_name}")

    # 如果你想查看所有可用的GPU设备及其名字
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
