# Virtual Stylist (King's College London - BSc Computer Science)

![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat-square&logo=OpenCV&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/-Numpy-013243?style=flat-square&logo=NumPy)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-000000?style=flat-square&logo=python)

## Overview
For my final-year project, I was lucky to have the opportunity to build whatever system I wanted. From the start of my degree, I knew that I wanted to build something that incorporated my love for fashion, growing interest in Machine Learning and, challenged my system design skills. Considering this was a relatively unexplored intersection between fashion and tech, I decided to create a virtual stylist. 

## Approach
* Runway images from [Vogue Runway](https://www.vogue.com/fashion-shows/latest-shows) and individual clothing item images from [Farfetch](https://www.farfetch.com/uk/?utm_source=google&utm_medium=cpc&utm_keywordid=58953085&isbrand=yes&pid=google_search&af_channel=Search&c=61800696&af_c_id=61800696&af_siteid=&af_keywords=aud-2324821770157:kwd-533010193&af_adset_id=2867078016&af_ad_id=591291745084&af_sub1=58953085&is_retargeting=true&gad_source=1&gad_campaignid=61800696&gbraid=0AAAAADsmKHQu-7XlqoqfcqeorAvRwBcLx&gclid=CjwKCAjwiY_GBhBEEiwAFaghvl0njCIrWVKcBl8tMVhjmhHJlcvwaK65xrR8l1JsgnSsyMJYGi2kbhoCwU4QAvD_BwE) as **training data**
  * Images were collated **manually** to avoid violating copyright protections on images
  
* Feature extraction from **training data**
  *  **Silhouette** of outfits and clothing items extracted using [**Hu Moments**](#hu-moments)
  *  Top 3 most **prominent colours** in images extracted using [**K-Means Clustering**](#k-means-clustering)
  *  Runway **show date** extracted from image **metadata** using `xattr` library

* **Training data** images and features stored in `.csv` files using [`CSVCreator`](./csv_creator.py) and `Pandas` library
  * Runway images stored in outfit csv with coloumns: image path, prominent colours, show date
  * Clothing item images stored in clothing item csv with columns: image path, item **label** (e.g. skirt, top, jacket, etc.), prominent colours

* [`Outfit Assembler`](./outfit_assembler.py) 


### K-Means Clustering

## Further Improvements


