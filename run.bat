cd server_api
docker build -t toxic-metr-app .
docker run -p 8000:8000 toxic-metr-app