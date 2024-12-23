import os
import re

def append_to_bench_file(directory, filename, content):
    """
    在指定文件夹下创建一个以.bench结尾的文件，如果文件已存在，则在该文件上追加内容。

    参数:
    directory (str): 文件将要创建的目录。
    filename (str): 文件的名字（不包括.bench扩展名）。
    content (str): 要写入文件的内容。
    """
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)
    
    # 创建文件的完整路径
    file_path = os.path.join(directory, f"{filename}.bench")
    
    # 检查文件是否已存在
    if os.path.exists(file_path):
        # 文件存在，追加内容
        with open(file_path, 'a') as file:  # 使用'a'模式打开文件，用于追加内容
            file.write(content)
            #file.write('\n')  # 可以选择在每次追加内容后添加一个换行符
    else:
        # 文件不存在，创建新文件并写入内容
        with open(file_path, 'w') as file:
            file.write(content)


def replace_nodes(file_content, dic_nodes):
    # 构建正则表达式模式，匹配以 n, po, 或 a 开头后跟一个或多个数字的模式
    pattern = r'\bn(\d+)|\bpo(\d+)|\ba(\d+)'
    
    for i, line in enumerate(file_content):
        # 检查元素是否为字符串
        if isinstance(line, str):
            # 使用正则表达式替换行中的所有匹配键实例
            def replace(match):
                # 获取匹配的完整键
                full_key = match.group(0)
                # 返回字典中对应的值，如果键不存在则返回原始键
                return str(dic_nodes.get(full_key, full_key))
            
            # 替换行中的所有匹配键实例
            file_content[i] = re.sub(pattern, replace, line)




def extract_lines_from_files(directory, aimed_directory):
    """
    读取给定目录下每个文件的内容，并提取每行的信息。

    参数:
    directory (str): 要读取的目录路径。

    返回:
    list: 包含所有文件每行信息的列表。
    """
    dic = {"0xe8": [0, 0, 0], "0x8e":[1, 0, 0], "0xb2":[0, 1, 0], "0xd4":[0, 0, 1]}

    for filename in os.listdir(directory):
        file_content = []
        added_node_dic = []
        added_nodes = 0
        count_input = 1
        count_output = 0
        count_inner_index = 0
        total_nodes = 0
        dic_nodes = {}
        file_path = os.path.join(directory, filename)
        aimed_file_path = os.path.join(aimed_directory, f"{filename}.bench")
        if os.path.isfile(file_path):  # 确保是文件
            with open(file_path, 'r') as file:  # 打开文件
                dic_nodes["n0"] = 0
                lines = file.readlines()  # 读取所有行
                file_content.append("INPUT(n0)\n")
                for line in lines:
                    if "LUT" not in line:
                        file_content.append(line)
                        if "INPUT" in line:
                            input = line.split("(")[1].split(")")[0]
                            dic_nodes[input] = count_input
                            count_input += 1
                        elif "OUTPUT" in line:
                            output = line.split("(")[1].split(")")[0]
                            dic_nodes[output] = count_output + count_input 
                            count_output += 1
                        #append_to_bench_file(aimed_directory, filename, line)
                    else:
                        count_inner_index += 1
                        if "po" not in line:
                            inner_node = line.split(" ")[0]
                            #print("inner_node =", inner_node)
                            dic_nodes[inner_node] = count_output + count_input + count_inner_index - 1
                        if "0xe8" in line:
                            content = line.split(" ")
                            file_content.append(content[0] + " = " + "MAJ" + content[4] + " " + content[5] + " " + content[6])
                        elif "0x8e" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "MAJ" + "(" + added_node_dic[added_nodes] + "," + " " + content[5] + " " + content[6])
                            node = line.split("(")[1].split(",")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0xb2" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "MAJ"  + content[4] + " " + added_node_dic[added_nodes] + "," + " " + content[6])
                            node = line.split("(")[1].split(",")[1].split(" ")[1]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0xd4" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "MAJ"  + content[4] + " " + content[5]  + " " + added_node_dic[added_nodes] + ")" + '\n')
                            node = line.split("(")[1].split(" ")[2].split(")")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0x2" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "BUF"  + content[4])
                            added_nodes += 1
                        elif "0x1" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "NOT"  + content[4])
                            added_nodes += 1
                total_nodes = count_input + count_inner_index + 1
                #print("total_nodes =", total_nodes)
                for i in range(added_nodes):
                    dic_nodes[f"a{i}"] = total_nodes - 1
                    total_nodes += 1
                #print("dic_nodes =", dic_nodes)
                #print("file_content =", file_content)
                replace_nodes(file_content, dic_nodes)
                with open(aimed_file_path, 'w') as file:
                    for item in file_content:
                        if item != '0 = gnd\n':
                            file.write(item)




                            

if __name__ == "__main__":
# 使用示例
    directory = '/home/jwt/raw1_mig'
    aimed_directory = '/home/jwt/aimed_mig'
    extracted_lines = extract_lines_from_files(directory, aimed_directory)

