import logging
import sys
import os 
import datetime 

abs_path = os.path.abspath('.')

file_path = f"{abs_path}/log"
file_name = f"{datetime.datetime.now().strftime('%Y_%m_%d')}.log"
file_path_name = f"{file_path}/{file_name}"

print(file_path, file_name, file_path_name)
if not os.path.isdir(file_path):
    os.mkdir(file_path)

if os.path.isfile(file_path_name):
    os.remove(file_path_name)
    
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

#format= logging.Formatter('%(asctime)s [ %(levelname)s ] - %(filename)s : %(lineno)s >> %(message)s')
#format= logging.Formatter('%(message)s')
format= logging.Formatter('%(asctime)s [ %(levelname)s ] - %(message)s')

stream_handler = logging.StreamHandler()

file_handler = logging.FileHandler(file_path_name, encoding="utf-8")

stream_handler.setFormatter(format)
file_handler.setFormatter(format)

#logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def logger_exception(ex, memo=""):
    exc_space = " "*24
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logger.warn(f"\n{exc_space}exc_type = {exc_type}\n{exc_space}file_name = {file_name}\n{exc_space}tb_lineno = {exc_tb.tb_lineno}\n{exc_space}exc_obj = {exc_obj}\nmemo={memo}")
