from abc import ABCMeta, abstractmethod

class AbstractTaskCase():

    '''1. Read the folder information
        json 파일, image 파일 정보 읽기
    '''
    @abstractmethod
    def set_data():
        pass

    '''2. Resize the file
        모든 image 파일 resize
    '''
    @abstractmethod
    def resize():
        pass

    # '''3. Write image and json'''
    # @abstractmethod
    # def save():
    #     pass

    '''4. View segmentation&bbox'''
    @abstractmethod
    def view():
        pass

    '''Full logic'''
    @abstractmethod
    def run():
        pass