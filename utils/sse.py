import json
import os
import sys
import glob
from pathlib import Path
import numpy as np



def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 将数据转成 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message, flush=True)

# sse_print("start", {"status": "success", "message": "SSE started."})
def sse_input_path_validated(args):
    try:
        if os.path.exists(args.input_path):
            sse_print("input_path_validated", {
                "status": "success",
                "message": "Input path is valid and complete.",
                "file_name": args.input_path
            })
            
            try:
                if os.path.exists(f'{args.input_path}/samples'):
                    data_files = glob.glob(os.path.join(f'{args.input_path}/samples', '*/'))
                    sse_print("input_data_validated", {
                        "status": "success",
                        "message": "Input data file is valid and complete.",
                        "file_name": data_files[0] if data_files else f'{args.input_path}/samples'
                    })
                else:
                    raise ValueError('Input data file not found.')
            except Exception as e:
                sse_print("input_data_validated", {"status": "failure", "message": f"{e}"})
                
            try:
                if os.path.exists(f'{args.checkpoint}'):
                    model_files = glob.glob(os.path.join(f'{args.checkpoint}', '*'))
                    sse_print("input_model_validated", {
                        "status": "success",
                        "message": "Input model file is valid and complete.",
                        "file_name": model_files[0] if model_files else f'{args.checkpoint}'
                    })
                else:
                    raise ValueError('Input model vad validate.')
            except Exception as e:
                sse_print("input_model_validated", {"status": "failure", "message": f"{e}"})
        else:
            raise ValueError('Input path not found.')
    except Exception as e:
        sse_print("input_path_validated", {"status": "failure", "message": f"{e}"})

def sse_output_path_validated(args):
    try:
        if os.path.exists(args.output_path):
            sse_print("output_path_validated", {
                "status": "success",
                "message": "Output path is valid and complete.",
                "file_name": args.output_path
            })
        else:
            raise ValueError('Output path not found.')
    except Exception as e:
        sse_print("output_path_validated", {"status": "failure", "message": f"{e}"})

def sse_adv_samples_gen_validated(adv_image_name):
    sse_print("adv_samples_gen_validated", {
        "status": "success",
        "message": "adv sample is generated.",
        "file_name": adv_image_name
    })

def sse_clean_samples_gen_validated(clean_image_name):
    sse_print("clean_samples_gen_validated", {
        "status": "success",
        "message": "clean sample is generated.",
        "file_name": clean_image_name
    })

def sse_epoch_progress(progress, total, epoch_type="Epoch"):
    sse_print("training_progress", {
        "progress": progress,
        "total": total,
        "type": epoch_type
    })

def sse_error(message, event_name="error"):
    sse_print(event_name, {"status": "failure", "message": message})

