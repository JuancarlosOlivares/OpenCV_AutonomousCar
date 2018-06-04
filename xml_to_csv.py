import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'): #find a xml files inside the folders
        tree = ET.parse(xml_file)
        root = tree.getroot()
        #open up xml file and get the data such as:
        #filename, width, height, xmin, ymin, xmax, and ymax
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     #member[0].text,
                     int(member[5][0].text),
                     int(member[5][1].text),
                     int(member[5][2].text),
                     int(member[5][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
    #put data inside a data frame in order to print in a csv file.
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in ['train', 'test']: #look inside train and test directory
        image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
        #image_path will get the image paths from either train or test
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')


main()
