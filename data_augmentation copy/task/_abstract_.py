from abc import ABCMeta, abstractmethod

class AbstractTaskCase():

    '''1. Read the folder information
        json 파일, image 파일 정보 읽기
    '''
    @abstractmethod
    def set_folder_data():
        pass
    
    '''2. Read the file(.png) information
        ???
    # '''
    # @abstractmethod
    # def set_file_data():
    #     pass

    # '''3. Resize the file'''
    # @abstractmethod
    # def resize():
    #     pass

    # '''4. View segmentation&bbox'''
    # @abstractmethod
    # def view():
    #     pass

    '''Full logic'''
    @abstractmethod
    def run():
        pass