from library.utils.header import *
from library.utils.io import *

from library.utils.log import logger, logger_exception

from library.config import view_configs as vw_cfg

def cv_view(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plt_view(image):
    plt.imshow(image)
    plt.show()
    
class View:
    def __init__(self):
        super().__init__()
        self.box_color = self._hex_to_rgb(vw_cfg["view"]["box_color"])
        self.text_color = self._hex_to_rgb(vw_cfg["view"]["text_color"])
        
        self.font_path = vw_cfg["view"]["font_path"]
        self.text_size = vw_cfg["view"]["text_size"]
        
        self.is_image_save = vw_cfg["save"]["is_image_save"]
        self.is_image_one_by_annotation_save = vw_cfg["save"]["is_image_one_by_annotation_save"]
        
        self.is_bbox = vw_cfg["save"]["is_bbox"]
        self.is_segmentation = vw_cfg["save"]["is_segmentation"]
        self.is_title = vw_cfg["save"]["is_title"]
        
        self.is_clear = vw_cfg["save"]["is_clear"]
        
        self.font = ImageFont.truetype(self.font_path, self.text_size)
        
    def _rgb_to_hex(self, RGB_color):        
        return '#'+hex(RGB_color[0])[2:]+hex(RGB_color[1])[2:]+hex(RGB_color[2])[2:]

    def _hex_to_rgb(self, hex_code):
        index = (0,2,4)
        if "#" in hex_code:
            index = (1,3,5)
        return np.array([ int(hex_code[i:i + 2], 16) for i in index ], dtype=np.float32)

    def _mask(self, seg, width, height):
        mask = np.zeros((height, width))
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        result = cv2.fillPoly(mask, [poly], 1).astype(bool)
        return result
    
    def _bbox(self, bbox, width, height):
        mask = np.zeros((height, width))
        bbox = [ 
            int(bbox[0]), int(bbox[1]), 
            int(bbox[0]), int(bbox[3]+bbox[1]), 
            int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1]),
            int(bbox[2]+bbox[0]), int(bbox[1])
        ]
        
        poly = np.array(bbox).reshape((int(len(bbox)/2), 2))
        result = cv2.polylines(mask, [poly], True, (255,255,255), 1).astype(bool)
        return result
    
    def _label(self, image_data, label, width, height, x, y, text_size):
        mask = np.zeros((height, width))
        img_pil = Image.fromarray(mask)
        draw = ImageDraw.Draw(img_pil)
        
        draw.text((x, y), label, font=self.font, stroke_width=2, stroke_fill="black")
        result = np.array(img_pil).astype(bool)
        return result
    
    def _convert_title_point(self, bbox, text_size):
        x = int(bbox[0])
        y = int(bbox[1]-text_size)
        width = int(bbox[2]+x)
        height =int(bbox[1])
        
        return [ 
            x,y, 
            width,y, 
            width,height,
            x,height,
        ]
    
    def _is_image_save(self, image_data, save_path, save_name):
        if self.is_image_save:
            cv_save_image(image_data, save_path, save_name, np.uint8, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    
    def _is_image_one_by_annotation_create(self, image_data):
        if  self.is_image_one_by_annotation_save:
            return copy.copy(image_data)
        else:
            return None
            
    def _is_image_one_by_annotation_grid(self, image_data, mask, alpha, color, beta):
        if  self.is_image_one_by_annotation_save:
            return self._grid_mask(image_data, mask, alpha, color, beta)
        else:
            return None
        
    def _is_image_one_by_annotation_save(self, image_data, save_path, save_name, index, label):
        if  self.is_image_one_by_annotation_save:
            path = f"{save_path}/{split_extension_file_name(save_name)}"
            file_name = f"{index}_{label}_{save_name}"
            
            check_dir(path)
            cv_save_image(image_data, path, file_name, np.uint8, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    
    def _grid_mask(self, image_data, mask, alpha, color, beta):
        image_data[mask] = image_data[mask] * alpha + color * beta
        return image_data
    
    def visualize(self, save_path, save_name, image_data, json_data):
        
        images = json_data["images"]
        categories = json_data["categories"]
        annotations = json_data["annotations"]
        
        for image in images:
            grid_image_data = image_data.astype(np.float32)
        
            width = image["width"]
            height = image["height"]
            
            for index, annotation in enumerate(annotations):
                grid_annotation_data = self._is_image_one_by_annotation_create(image_data.astype(np.float32))
                
                segmentation = annotation["segmentation"]
                bbox = annotation["bbox"]
                label = categories[annotation["category_id"]-1]["name"]
                color = self._hex_to_rgb(categories[annotation["category_id"]-1]["color"])
                score = annotation['score']
                
                #grid segmentation
                if self.is_segmentation:
                    for seg in segmentation:
                        grid = self._mask(seg, width, height)
                        grid_image_data = self._grid_mask(grid_image_data, grid, 0.5, color, 0.5)
                        grid_annotation_data = self._is_image_one_by_annotation_grid(grid_annotation_data, grid, 0.5, color, 0.5)

                #grid bbox
                if self.is_bbox:
                    grid = self._bbox(bbox, width, height)
                    grid_image_data = self._grid_mask(grid_image_data,grid, 0.5, self.box_color, 0.5)
                    grid_annotation_data = self._is_image_one_by_annotation_grid(grid_annotation_data, grid, 0.5, self.box_color, 0.5)
                
                #title
                if self.is_title:
                    title_point = self._convert_title_point(bbox, self.text_size)
                    
                    grid = self._mask(title_point,width, height)
                    grid_image_data = self._grid_mask(grid_image_data,grid, 0.5, self.box_color, 0.5)
                    grid_annotation_data = self._is_image_one_by_annotation_grid(grid_annotation_data, grid, 0.5, self.box_color, 0.5)
                    
                    title = str(f"{label} : {np.round(score,3)}")
                    grid = self._label(
                        grid_image_data, 
                        title,
                        width,
                        height,
                        title_point[0],
                        title_point[1], 
                        self.text_size
                    )
                    grid_image_data = self._grid_mask(grid_image_data,grid, 0, self.text_color, 1)
                    grid_annotation_data = self._is_image_one_by_annotation_grid(grid_annotation_data, grid, 0, self.text_color, 1)
                    
                self._is_image_one_by_annotation_save(grid_annotation_data, save_path, save_name, index, label)
                
            self._is_image_save(grid_image_data, save_path, save_name)   
        