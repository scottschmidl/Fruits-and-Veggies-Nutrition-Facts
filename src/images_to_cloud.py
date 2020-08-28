# pushing to aws, mvp+
import boto3
import os
from EDA_img import get_file_names

def print_s3_contents_boto3(connection):
    ''''borrowed from DSI AWS lecture. send images to cloud'''
    for bucket in connection.buckets.all():
        for key in bucket.objects.all():
            print(key.key)

if __name__ == '__main__':
    #path to where all imgs are stored localy
    directory_str = '../../../../data/img/img_dumps/' #change this
    print(directory_str)
    bucket_name = 'faster-pet-adoption.bucket' #change this
    file_names = get_file_names(directory_str)
    print(file_names)
    print("\nStarted boto3 connection") 
    boto3_connection = boto3.resource('s3')
    print_s3_contents_boto3(boto3_connection)
    s3_client = boto3.client('s3')
    for file_name in file_names:
        local_file = directory_str + file_name
        s3_client.upload_file(local_file, bucket_name, file_name)
        print(f"\nUploaded file {file_name}")
        print_s3_contents_boto3(boto3_connection)