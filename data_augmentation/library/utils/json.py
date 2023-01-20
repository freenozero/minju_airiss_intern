from library.utils.header import json as js

class json:
    '''json load, dump'''

    def load_json(json_path):
        '''Load json file'''

        with open(json_path) as json_file:
            json_data = js.load(json_file)
        
        json_file_dic = {}
        # file_name = i 형식
        for i in range(len(json_data['images'])):
            file_name = json_data['images'][i]['file_name']
            json_file_dic[file_name] = i
            
        return json_data, json_file_dic
    
    
    def dump_json(json_path, json_data):
        '''Dump json data'''
        with open(json_path, 'w') as json_file:
            js.dump(json_data, json_file)