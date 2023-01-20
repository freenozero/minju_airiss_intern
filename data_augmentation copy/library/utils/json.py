from library.utils.header import json as js

class json:
    '''json load, dump'''

    def load_json(json_path):
        '''Load json file'''
        with open(json_path) as json_file:
            json_data = js.load(json_file)
        return json_data
    
    
    def dump_json(json_path, json_data):
        '''Dump json data'''
        with open(json_path, 'w') as json_file:
            js.dump(json_data, json_file)