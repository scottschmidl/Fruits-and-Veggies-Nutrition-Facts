import boto3
import glob
import os

## PRINT OUT BUCKET NAMES
# s3 = boto3.resource('s3')
# for bucket in s3.buckets.all():
#     print(bucket.name)

## MULTIPART UPLOAD(NOT EXACTLY SURE HOW TO DO THIS)
# object_key = 'health'
# id = '1234'
# multipart_upload = s3.MultipartUpload(bucket_name=bucket_name, object_key=object_key, id=id)

def upload_files(path):

    s3 = boto3.Session().resource('s3')
    bucket = s3.Bucket('fruitsveggiesimages')
    i = 0
    for subdir, _, filess in os.walk(path):
        for files in filess:
            full_path = os.path.join(subdir, files)
            with open(full_path, 'rb') as data:
                bucket.put_object(Key=full_path[len(path)+1:], Body=data)
                i += 1
                if i % 6563 == 0:
                    print(f'Completed Uploading Image: {i} of 78756!\n')

def main():

    path = '../data/Train'
    upload_files(path)
    print('\nCompleted Upload of Images!')

if __name__ == '__main__':
    main()