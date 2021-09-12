import pandas as pd
import boto3
import os


s3 = boto3.resource(service_name='s3',
                    region_name='us-east-2',
                    aws_access_key_id='mykey',
                    aws_secret_access_key='mysecretkey')

for bucket in s3.buckets.all():
    print(bucket.name)

os.environ["AWS_DEFAULT_REGION"] = 'us-east-2'
os.environ["AWS_ACCESS_KEY_ID"] = 'mykey'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'mysecretkey'

for obj in s3.Bucket('cheez-willikers').objects.all():
    print(obj)

obj = s3.Bucket('cheez-willikers').Object('foo.csv').get()
foo = pd.read_csv(obj['Body'], index_col=0)
