#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型輕量化工具
包含onnxslim優化和float32轉float16
"""

import os
import argparse
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
import onnxslim

def get_model_size(model_path):
    """獲取模型檔案大小（MB）"""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)

def optimize_with_onnxslim(input_path, output_path):
    """使用onnxslim進行模型優化"""
    print(f"正在使用onnxslim優化模型: {input_path}")
    
    try:
        # 使用onnxslim進行模型優化
        onnxslim.slim(input_path, output_path)
        print(f"onnxslim優化完成，輸出至: {output_path}")
        return True
    except Exception as e:
        print(f"onnxslim優化失敗: {str(e)}")
        return False

def fix_cast_node_types(model):
    """
    修復模型中Cast節點的類型不匹配問題
    """
    import onnx.helper as helper
    
    graph = model.graph
    
    # 遍歷所有節點，找到有問題的Cast節點
    nodes_to_remove = []
    nodes_to_add = []
    
    for i, node in enumerate(graph.node):
        if node.op_type == 'Cast':
            # 檢查Cast節點的to屬性
            to_type = None
            for attr in node.attribute:
                if attr.name == 'to':
                    to_type = attr.i
                    break
            
            # 如果Cast到float16但下游期望float32，則移除或修改
            if to_type == onnx.TensorProto.FLOAT16:
                print(f"發現可能有問題的Cast節點: {node.name}")
                # 可以選擇移除這個Cast節點或修改其類型
                # 這裡我們選擇將其修改為Cast到float32
                new_node = onnx.helper.make_node(
                    'Cast',
                    inputs=node.input,
                    outputs=node.output,
                    to=onnx.TensorProto.FLOAT,
                    name=node.name
                )
                nodes_to_remove.append(i)
                nodes_to_add.append(new_node)
    
    # 移除舊節點並添加新節點
    new_nodes = []
    for i, node in enumerate(graph.node):
        if i not in nodes_to_remove:
            new_nodes.append(node)
    
    new_nodes.extend(nodes_to_add)
    
    # 重建圖
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    return model

def convert_to_fp16(input_path, output_path):
    """將float32轉換為float16"""
    print(f"正在將模型轉換為float16: {input_path}")
    
    try:

        # 載入模型
        model = onnx.load(input_path)
        # 轉換為float16
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True, op_block_list=["Cast"])
        #model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False,
                         #disable_shape_infer=False, op_block_list=None, node_block_list=None)
        onnx.save(model_fp16, output_path)
        '''
        # 驗證模型結構
        try:
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3  # 只顯示錯誤
            ort.InferenceSession(output_path, session_options)
            print(f"✅ 模型載入成功")
            print("✓ 模型結構檢查通過")
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            return False
        '''
        # 儲存轉換後的模型
        print(f"float16轉換完成，輸出至: {output_path}")
        return True
    except Exception as e:
        print(f"float16轉換失敗: {str(e)}")
        return False

def optimize_onnx_model(input_path, output_dir=None, use_onnxslim=True, use_fp16=True):
    """
    完整的ONNX模型輕量化流程
    
    Args:
        input_path (str): 輸入ONNX模型路徑
        output_dir (str): 輸出目錄，如果為None則使用輸入檔案所在目錄
        use_onnxslim (bool): 是否使用onnxslim優化
        use_fp16 (bool): 是否轉換為float16
    """
    
    if not os.path.exists(input_path):
        print(f"錯誤: 找不到輸入檔案 {input_path}")
        return
    
    # 設定輸出目錄
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 獲取原始檔案資訊
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    original_size = get_model_size(input_path)
    print(f"原始模型大小: {original_size:.2f} MB")
    
    current_model = input_path
    
    # 步驟1: onnxslim優化
    if use_onnxslim:
        slim_output = os.path.join(output_dir, f"{base_name}_slim.onnx")
        if optimize_with_onnxslim(current_model, slim_output):
            slim_size = get_model_size(slim_output)
            print(f"onnxslim優化後大小: {slim_size:.2f} MB (減少 {((original_size - slim_size) / original_size * 100):.1f}%)")
            current_model = slim_output
        else:
            print("onnxslim優化失敗，跳過此步驟")
    
    # 步驟2: float16轉換
    if use_fp16:
        if use_onnxslim:
            fp16_output = os.path.join(output_dir, f"{base_name}_slim_fp16.onnx")
        else:
            fp16_output = os.path.join(output_dir, f"{base_name}_fp16.onnx")
        
        if convert_to_fp16(current_model, fp16_output):
            fp16_size = get_model_size(fp16_output)
            print(f"float16轉換後大小: {fp16_size:.2f} MB (總共減少 {((original_size - fp16_size) / original_size * 100):.1f}%)")
            current_model = fp16_output
        else:
            print("float16轉換失敗，跳過此步驟")
    
    # 最終結果
    if current_model != input_path:
        final_size = get_model_size(current_model)
        compression_ratio = (original_size - final_size) / original_size * 100
        print(f"\n✅ 優化完成!")
        print(f"原始大小: {original_size:.2f} MB")
        print(f"優化後大小: {final_size:.2f} MB")
        print(f"壓縮比例: {compression_ratio:.1f}%")
        print(f"輸出檔案: {current_model}")
    else:
        print("\n❌ 未進行任何優化")

def main():
    parser = argparse.ArgumentParser(description='ONNX模型輕量化工具')
    parser.add_argument('-input', default='./model/AIF2.0/reconstructed_model.onnx',help='輸入ONNX模型路徑')
    parser.add_argument('-o', '--output', default='./model/AIF2.0/slim_model.onnx', help='輸出目錄 (預設: 與輸入檔案同目錄)')
    parser.add_argument('--no-slim',default=False, help='跳過onnxslim優化')
    parser.add_argument('--no-fp16',default=False, help='跳過float16轉換')
    
    args = parser.parse_args()
    
    optimize_onnx_model(
        input_path=args.input,
        output_dir=args.output,
        use_onnxslim=not args.no_slim,
        use_fp16=not args.no_fp16
    )

if __name__ == "__main__":
    # 如果直接執行，可以在這裡設定預設參數進行測試
    import sys
    main()
