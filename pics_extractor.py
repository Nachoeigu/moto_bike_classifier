#We will use Mercado Libre as a source of images already classified.
import requests
from lxml import html
import shutil
from constants import BASIC_HEADER, BIKE_SUBCATEGORIES, MOTORCYCLE_SUBCATEGORIES
from utils import count_files, get_image, save_file
import numpy as np

class MercadoLibre:

    def __init__(self):
        self.motorcycle_links = []
        self.bike_links = []

    def __get_motorcycles_img_urls(self):
        #We should scrape the results of each subcategory so we can achieve more examples
        for subcategory in MOTORCYCLE_SUBCATEGORIES:
            for number in range(1,4000, 50):
                response = requests.get(f'https://motos.mercadolibre.com.ar/{subcategory}/motocicleta_Desde_{number}_NoIndex_True', headers=BASIC_HEADER)
                if response.status_code != 200:
                    break

                data = html.fromstring(response.content)
                urls = data.xpath("//ol[contains(@class,'search')]//li//img/@data-src")
                for url in urls:
                    if url in self.motorcycle_links:
                        pass
                    else:
                        self.motorcycle_links.append(url.replace("webp","jpg"))

    def get_imgs_motorcycle(self):
        self.__get_motorcycles_img_urls()

        for index, url in enumerate(self.motorcycle_links):
            print("We are downloading the picture about motorcycles number ", index + 1)
            image = get_image(url)
            save_file(image, 'data/train/motorcycle',index)
            
    def __get_bikes_img_urls(self):
        for subcategory in BIKE_SUBCATEGORIES:
            for number in range(1,4000, 50):
                response = requests.get(f'https://listado.mercadolibre.com.ar{subcategory}bicicleta_Desde_{number}_NoIndex_True', headers=BASIC_HEADER)
                if response.status_code != 200:
                    break

                data = html.fromstring(response.content)
                urls = data.xpath("//ol[contains(@class,'search')]//li//img/@data-src")
                for url in urls:
                    if url in self.bike_links:
                        pass
                    else:
                        self.bike_links.append(url.replace("webp","jpg"))

    def get_imgs_bike(self):
        self.__get_bikes_img_urls()

        for index, url in enumerate(self.bike_links):
            print("We are downloading the picture about bikes number ", index + 1)
            image = get_image(url)
            save_file(image, 'data/train/bike',index)
                    
    def creating_testing_dataset(self):
        total_bike_imgs, total_motorcycle_imgs = count_files()

        random_bike_pics_to_testing = list(set(list(np.random.randint(0, total_bike_imgs, 600))) )
        random_motorcycle_pics_to_testing = list(set(list(np.random.randint(0, total_motorcycle_imgs, 600))))
        
        for number in random_bike_pics_to_testing:
            shutil.move(f'data/train/bike/{number}.jpg', f'data/test/bike/{number}.jpg')
        
        for number in random_motorcycle_pics_to_testing:
            shutil.move(f'data/train/motorcycle/{number}.jpg', f'data/test/motorcycle/{number}.jpg')
    
