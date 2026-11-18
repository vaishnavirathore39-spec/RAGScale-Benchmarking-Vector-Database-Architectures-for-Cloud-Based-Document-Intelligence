
import os
import boto3

from botocore.exceptions import ClientError, NoCredentialsError


class S3Storage:
    """Access an existing private S3 bucket."""

    def __init__(self, bucket_name=None, region=None):

        self.bucket_name = (
            bucket_name or os.getenv("AWS_S3_BUCKET")
        )

        self.region = (
            region or os.getenv("AWS_REGION", "ap-southeast-2")
        )

        if not self.bucket_name:
            raise ValueError(
                "AWS_S3_BUCKET is not configured"
            )

        session = boto3.session.Session(
            region_name=self.region
        )

        self.client = session.client("s3")

    def list_objects(self, prefix=""):
        """List objects in an existing bucket."""

        try:
            paginator = self.client.get_paginator(
                "list_objects_v2"
            )

            keys = []

            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix
            ):
                for item in page.get("Contents", []):
                    keys.append(item["Key"])

            return keys

        except (ClientError, NoCredentialsError) as exc:
            raise RuntimeError(
                f"S3 listing failed: {exc}"
            ) from exc

    def read_object(self, key):
        """Read an S3 object as bytes."""

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )

            return response["Body"].read()

        except (ClientError, NoCredentialsError) as exc:
            raise RuntimeError(
                f"S3 download failed: {exc}"
            ) from exc

    def read_text(self, key, encoding="utf-8"):
        """Read an S3 object as text."""

        return self.read_object(key).decode(encoding)
