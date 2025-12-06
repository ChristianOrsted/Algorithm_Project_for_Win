import os
map_dir = r"D:\Program Files (x86)\StarCraft II\Maps\SMAC_Maps"
if os.path.exists(map_dir):
    print("SMAC_Maps 文件夹存在")
    print("地图文件：", os.listdir(map_dir))
else:
    print("SMAC_Maps 文件夹不存在")