import datetime
import traceback
import logging

def trace_e(e):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 使用traceback获取详细的错误跟踪信息
    error_traceback = ''.join(traceback.format_tb(e.__traceback__))
    
    # 打印详细错误日志
    return (f"\n{now} - 发生了一个错误：\n"
            f"异常类型: {type(e).__name__}\n"
            f"错误信息: {e}\n"
            f"发生位置:\n{error_traceback}\n")

def set_logger(level):

    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(level)  # 设置日志级别

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler('my_log.log')
    fh.setLevel(level)

    # 再创建一个handler，用于输出到控制台
    # ch = logging.StreamHandler()
    # ch.setLevel(level)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    # logger.addHandler(ch)
    return logger