#coding = utf-8


import os

#从文件的根目录读取根目录下的所有文件
def gen_rule_set(rule_file_root):
    rule_files = []
    rule_files_name = os.listdir(rule_file_root)
    for file in rule_files_name:
        if not os.path.isdir(os.path.join(rule_file_root, file)):
            rule_files.append(file)  # 只添加文件的名称
    return rule_files


# if __name__ == '__main__':
#     rule_file_root = "/home/zjc/test_tuningPLM/EXER/"
#     test_gen = gen_rule_set(rule_file_root)
#     print("看看会出现什么",test_gen)
