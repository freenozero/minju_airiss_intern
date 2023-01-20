from abc import ABCMeta, abstractmethod

class AbstractTaskCase():

    @abstractmethod
    def set_data():
        '''1. Read the folder information'''
        pass

    @abstractmethod
    def resize():
        '''2. Resize the file'''
        pass

    @abstractmethod
    def view():
        '''3. View segmentation&bbox'''
        pass

    @abstractmethod
    def run():
        '''Full logic'''
        pass