import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_flipkart():
    api = []
    links = []
    url = "https://www.flipkart.com/clothing-and-accessories/topwear/tshirt/men-tshirt/pr?sid=clo,ash,ank,edy&otracker=categorytree&otracker=nmenu_sub_Men_0_T-Shirts"
    r = requests.get(url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    image_src = soup.find_all("img", {"class": "_2r_T1I"})

    soup2 = BeautifulSoup(htmlContent, 'html.parser')
    link_src = soup2.find_all("a", {"class": "_2UzuFa"})

    for ele in image_src:
        api.append(ele.get('src'))

    for ele in link_src:
        links.append('https://www.flipkart.com' + ele.get('href'))

    df = pd.DataFrame([api, links])
    df = df.T
    csv_filename = 'flipkart.csv'
    df.to_csv(csv_filename, index=False)


def get_vogue():
    api = []
    url = "https://www.vogue.in/vogue-closet/?closet=vogue_closet&filter_type=product_collection&order_by=recent&q=t+shirt&celebrity=&occasion=&price=&product-type=clothing"
    r = requests.get(url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    image_src = soup.find_all("div", {"class": "owl1-cols"})
    for ele in image_src:
        api.append(ele.find('img').get('src'))

    df = pd.DataFrame(api)
    csv_filename = 'vogue.csv'
    df.to_csv(csv_filename, index=False)


def get_pininterest():
    url = "https://www.pinterest.com/Marcellthekid/men-fashion-catalog/"
    img_class_name = "hCL kVc L4E MIw"
    api = []
    r = requests.get(url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    image_src = soup.find_all("img", {"class": img_class_name})
    for ele in image_src:
        api.append(ele.get('src'))

    df = pd.DataFrame(api)
    csv_filename = 'pininterest.csv'
    df.to_csv(csv_filename, index=False)


if __name__ == '__main__':
    get_flipkart()
    get_vogue()
    get_pininterest()
