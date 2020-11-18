#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Alejandro H.
# DATE CREATED:  9_8_2020                                 
# REVISED DATE: 11_10_2020
# PURPOSE: Create a function that retrieves the following 3 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Imports python modules
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args():
     # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 3 command line arguments args.dir for path to images files,
    # args.arch which CNN model to use for classification, args.labels path to
    # text file with names of dogs.
    parser.add_argument('--dir', type=str, default='pet_images/', 
                        help='path to folder of images')
    # TODO: 1a. EDIT parse.add_argument statements BELOW to add type & help for:
    #          --arch - the CNN model architecture
    #          --dogfile - text file of names of dog breeds
    parser.add_argument('--arch', type=str, default = 'vgg', help='classify_images.py' )
    parser.add_argument('--dogfile', type=str, default = 'dognames.txt', help='dognames.txt' )

    # TODO: 1b. Replace None with parser.parse_args() parsed argument 
    # collection that you created with this function 
    in_args= parser.parse_args()
    print("Argument 1",in_args.dir)
    print("Argument 1",in_args.arch)
    print("Argument 1",in_args.dogfile)
    return in_args

if __name__ == "__main__":
    get_input_args()



   
