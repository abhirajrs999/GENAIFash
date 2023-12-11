import requests
import shutil
import pandas as pd
import os 
from os import path

site_names = ['flipkart','pininterest','vogue']


for site_name in site_names:
    user_input1 = site_name + '.csv'

    directory = site_name

    img_path = os.path.join(directory) 
    if(path.isdir(img_path)==False):
        os.mkdir(img_path)
        print("Directory '% s' created" % directory) 

    df1 = pd.read_csv(user_input1)
    df_img_links = df1["0"]

    for i in range(len(df_img_links)):
        try:
            image_url = df_img_links[i]
            filename = str(i) + '.jpeg'

            # Open the url image, set stream to True, this will return the stream content.
            r = requests.get(image_url, stream = True)

            # Check if the image was retrieved successfully
            if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
                with open(os.path.join(img_path,filename),'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                
                print('Image sucessfully Downloaded: ',filename)
            else:
                print('Image Couldn\'t be retreived')
        except:
            pass