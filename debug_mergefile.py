import numpy as np

# 查看prob属性的内容
# 加载合并后的 .npz 文件
final_output_path = '/home/xqgrp/wangjingxin/datasets/mixgate_data/merged_all.npz'
merged_data = np.load(final_output_path, allow_pickle=True)['circuits'].item()

# 查看每个电路的数据。
for circuit_name, graph in merged_data.items():
    # 检查是否有 'aig_prob' 数据
    if 'xag_prob' in graph:
        print(f"\nCircuit: {circuit_name}")
        print(f"xag_prob: {graph['xag_prob']}")
    else:
        print(f"\nWarning: No 'mig_prob' data for circuit {circuit_name}")
