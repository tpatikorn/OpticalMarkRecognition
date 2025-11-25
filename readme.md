# About
It's an OMR

# Setup

1. For Google Login, create `.streamlit/secrets.toml` with

```
[auth]
redirect_uri = "http://[hostname]:[port]/[server.baseUrlPath]/oauth2callback"
client_id = "xxxxxxxxxxxxxx.apps.googleusercontent.com"
client_secret = "xxxxxxxxxxxxxx"
cookie_secret = "xxxxxxxxxxxxxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

2. install requirements `pip install -r requirements.txt`

3. to run, activate venv and `streamlit app.py --server.port=[port] --server.baseUrlPath=[prefix]"
