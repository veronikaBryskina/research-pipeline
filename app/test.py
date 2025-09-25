import fsspec

key = "nika"
secret = "p017g3RD2yuC"

endpoint_url = "https://aarav-nonmaterialistic-tesha.ngrok-free.dev"

fs = fsspec.filesystem(
    "s3",
    key=key,
    secret=secret,
    client_kwargs={"endpoint_url": endpoint_url},
    config_kwargs={"s3": {"addressing_style": "path"}} 
)

print(fs.ls(""))


