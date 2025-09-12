import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import biplist
import xattr
import osxmetadata
from feature_extractor import FeatureExtractor

class CSVCreator:

    def __init__(self):
        pass

    def create_outfit_csv(self):
        csv_frame = pd.DataFrame(columns=['Path', 'Prominent_Colour1', 'Prominent_Colour2', 'Prominent_Colour3', 'Silhouette', 'Presentation_Date'])

        outfits_path = os.path.expanduser("~/Desktop/stylist!/outfits")
                
        index = 0

        for outfit_folder in os.listdir(outfits_path):
            folder_path = os.path.join(outfits_path, outfit_folder)

            if os.path.isdir(folder_path):
                for image in os.listdir(folder_path):
                    index+=1
                    print(f'File {index}')

                    path = os.path.join(folder_path, image)
                    try:
                        colours = FeatureExtractor.prominent_colours_clustering(path)
                    except:
                        continue

                    silhouette = np.array(FeatureExtractor.get_silhouette(path), dtype=float)

                    attributes = xattr.xattr(path)
                    tags = attributes.get('com.apple.metadata:_kMDItemUserTags')
                    decoded_tag = biplist.readPlistFromString(tags)[0]
                    presentation_date = tuple(np.array(decoded_tag.split('_')).astype(int))

                    features_row = {
                        'Path': path,
                        'Prominent_Colour1': colours[0],
                        'Prominent_Colour2': colours[1],
                        'Prominent_Colour3': colours[2],
                        'Silhouette': silhouette,
                        'Presentation_Date': presentation_date
                    }

                    csv_frame = csv_frame._append(features_row, ignore_index=True)
        
        csv_frame.to_csv('outfits_csv.csv', index=True)

    def create_clothing_item_csv(self):
        csv_frame = pd.DataFrame(columns=['Path', 'Label', 'Prominent_Colour1', 'Prominent_Colour2', 'Prominent_Colour3'])

        clothing_item_path = os.path.expanduser("~/Desktop/stylist!/clothing_items")

        index = 0

        for clothing_item_folder in os.listdir(clothing_item_path):
            folder_path = os.path.join(clothing_item_path, clothing_item_folder)

            if os.path.isdir(folder_path):
                for image in os.listdir(folder_path):
                    index+=1
                    print(f'File {index}')

                    path = os.path.join(folder_path, image)
                    try:
                        colours = FeatureExtractor.prominent_colours_clustering(path)
                    except:
                        continue

                    while len(colours) < 3:
                        colours.append(None)

                    features_row = {
                        'Path': path,
                        'Label': clothing_item_folder,
                        'Prominent_Colour1': colours[0],
                        'Prominent_Colour2': colours[1],
                        'Prominent_Colour3': colours[2]
                    }

                    csv_frame = csv_frame._append(features_row, ignore_index=True)
        
        csv_frame.to_csv('clothing_items_csv.csv', index=True)
