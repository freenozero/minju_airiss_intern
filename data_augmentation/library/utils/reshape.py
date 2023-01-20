from library.utils.header import random, np
from library.utils.filesfolder import filesfolder

class reshape:
    '''image reshape 관련'''

    # 파일 첫번째가 짝수 and 현재 파일 자체가 짝수인지에 따라,
    # 파일 첫번째가 홀수 and 현재 파일 자체가 홀수인지에 따라,
    # random 숫자 돌리기
    def random(even, file_last_name, update_cols, update_rows):
        '''update_cols, update_rows'''
        if(even == filesfolder.even_distinction(file_last_name)):
            update_cols = round(random.uniform(0.5, 1.5), 1)
            update_rows = round(random.uniform(0.5, 1.5), 1)
        elif(even == filesfolder.even_distinction(file_last_name)):
            update_cols = round(random.uniform(0.5, 1.5), 1)
            update_rows = round(random.uniform(0.5, 1.5), 1)
        return update_cols, update_rows
    
    def return_segmentation(segmentation):
        '''segmentation array(shape: (?, 2)) return'''
        
        new_seg = list()
        for index, _ in enumerate(segmentation):
            if((index % 2) == 0):
                new_seg.append([segmentation[index], segmentation[index+1]])
        return np.array(new_seg, np.int32)
    
    def update_segmentation(segmentation, update_cols, update_rows):
        '''Update segmentation'''
        result = []
        for index, seg in enumerate(segmentation):
            if ((index % 2) == 0):
                result.append(round(seg * update_cols, 1))
            else:
                result.append(round(seg * update_rows, 1))   
        return [[result]]