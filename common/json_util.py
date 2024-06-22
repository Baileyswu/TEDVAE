import pandas as pd
import json
import logging
from numpy import generic
from common.log import trace_e

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, generic):
            return obj.item()  # 将Numpy类型转换为Python基础类型
        return super().default(obj)

def rewrite_json_to_csv(json_path, csv_path, desc=False):
    logging.info(f'convert {json_path} to {csv_path}')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')
    lines = [json.loads(x) for x in data if len(x) > 0]
    df = pd.DataFrame(lines)
    if desc:
        df = df.sort_index(ascending=False)
    df.to_csv(csv_path, index=False)
    return df

def parse_json(s):
    dt = {} 
    try:
        dt = json.loads(s)
    except Exception as e:
        logging.error(s)
        logging.error(trace_e(e))
    return dt