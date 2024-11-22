from pathlib import Path
import imghdr
import os
import shutil

# input 폴더는 원본 저장용으로 그대로 두고, 임의로 복사본을 생성해서 
# 해당 코드를 Run하는 것을 추천

data_dir = "./input_temp"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
            os.remove(filepath)
            if os.path.isfile(filepath):
                print("File still Exists") 
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
            os.remove(filepath)
            if os.path.isfile(filepath):
                print("File still Exists") 
        else:
            pass
