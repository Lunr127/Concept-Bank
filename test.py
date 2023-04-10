import os

# 打开文件并读取每一行内容
with open('tv22.avs.topics.txt', 'r') as file:
    lines = file.readlines()

# 创建一个空字典用于存储键值对
dictionary = {}

# 遍历每一行并将内容分割成键值对
for line in lines:
    key, value = line.strip().split('  ')
    dictionary[key] = value

# 检查 result 文件夹是否存在，如果不存在则创建
if not os.path.exists('result'):
    os.makedirs('result')

# 遍历字典并创建对应的文件
for key, value in dictionary.items():
    # 创建文件名为 key.txt 的文件并打开
    with open('result/'+key+'.txt', 'w') as file:
        # 将键值对写入文件中
        file.write(key + " " + value)

