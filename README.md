# Covid19 App - Installation with Docker

### Requirements
- Clone this repo
- You must install Docker and Docker-Compose

## Run command to get the code Backend and Weights
```bash
chmod +x installer.sh
./installer.sh
```

## Run command to run the application with Docker Compose
```bash
sudo docker-compose build
sudo docker-compose up
```

## Visit the url to test app database results
```http
http://localhost:1234 Databaseviewer
```

# To use the application, move the patient file to the /data/raw folder.
