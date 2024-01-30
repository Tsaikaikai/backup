要在Ubuntu系统中使用Python断开相机电源并重新启动USB相机，可以按照以下步骤进行操作：

安装必要的依赖库和工具：

sudo apt-get install python-pip
pip install pyudev
创建一个Python脚本，比如camera_restart.py，并将以下代码复制到该脚本中：

import pyudev
import subprocess

def monitor_callback(action, device):
    if action == "remove" and device.subsystem == "usb":
        for child in device.children:
            if child.subsystem == "video4linux":
                restart_camera()

def restart_camera():
    # 断电相机的命令
    subprocess.call(["sudo", "uhubctl", "-l", "1-1", "-p", "2", "-a", "0"])
    # 重启USB相机的命令
    subprocess.call(["sudo", "udevadm", "trigger", "--subsystem-match=video4linux"])

context = pyudev.Context()
monitor = pyudev.Monitor.from_netlink(context)
monitor.filter_by(subsystem='usb')
observer = pyudev.MonitorObserver(monitor, monitor_callback)
observer.start()

while True:
    try:
        monitor.poll(timeout=10)
    except KeyboardInterrupt:
        observer.stop()
        break
保存并退出脚本。

安装uhubctl命令行工具：

sudo apt-get install uhubctl
在终端中运行脚本：

python camera_restart.py
脚本将监视USB设备的插拔事件，并使用subprocess模块调用uhubctl命令断电相机，然后再通过udevadm命令触发重新启动USB相机。您可以根据需要自定义断电和重启相机的命令。
