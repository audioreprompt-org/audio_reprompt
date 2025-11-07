from __future__ import annotations

from typing import IO, Union
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


BytesLike = Union[bytes, IO[bytes]]


class S3Client:
    def __init__(self, bucket: str, region: str | None = None) -> None:
        self.bucket = bucket
        self._s3 = boto3.session.Session().client(
            "s3",
            region_name=region,
            config=Config(signature_version="s3v4"),
        )

    def upload_object(
        self,
        data: BytesLike,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        expires_in: int,
    ) -> str:
        """Uploads an in-memory object and returns a presigned GET URL."""
        try:
            extra_args = {"ContentType": content_type}
            if hasattr(data, "read"):
                self._s3.upload_fileobj(data, self.bucket, key, ExtraArgs=extra_args)
            else:
                self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
        except ClientError as e:
            raise RuntimeError(f"S3 upload failed: {e}") from e

        try:
            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to create presigned URL: {e}") from e
