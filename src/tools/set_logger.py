import time
import logging
import os

def set_logger(log_folder):
    """set logger"""
    assert log_folder is not None,"A log foler should be provoied to save log."
    logger = logging.getLogger("RAG WORK FLOW LOG")
    logger.setLevel(logging.DEBUG)
    cur_time = time.localtime()
    log_tag = "%s-%s-%s-%s-%s-%s"%(cur_time.tm_year,cur_time.tm_mon,cur_time.tm_mday,cur_time.tm_hour,cur_time.tm_min,cur_time.tm_sec)
    fh = logging.FileHandler(os.path.join(log_folder,log_tag))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger