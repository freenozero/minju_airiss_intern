import platform

def __font_path():
    #todo os별 위치 추가
    if "Windows" in platform.system():
        return "C:/Windows/Fonts/gulim.ttc"
    else:
        return "/usr/share/fonts/Ubuntu-R.ttf"

save = {
    "is_clear":True,
    "is_bbox":True,
    "is_title":True,
    "is_segmentation":True,
    
    "is_image_save":True,
    "is_image_one_by_annotation_save":True
}

view = {
    "font_path": __font_path(),
    "text_size":12,
    "box_color":"#ffffff",
    "text_color":"#000000"
}