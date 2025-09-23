import fsspec

key = ""
secret = ""

endpoint_url = "http://127.0.0.1:8089/"

fs = fsspec.filesystem(
    "s3",
    key=key,
    secret=secret,
    client_kwargs={"endpoint_url": endpoint_url},
    config_kwargs={"s3": {"addressing_style": "path"}} 
)

print(fs.ls(""))

with fs.open("mlflow/test.txt", "w") as f:
    f.write("Hello through Nginx /s3/!\n")


