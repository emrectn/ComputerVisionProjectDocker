# Covid19 App - Installation with Docker

We tried to create a web site which it can detect covid19 with deep learning methods.
### Requirements
- Clone this repo
- git submodule init
- git submodule update --remote
- You must install Docker and Docker-Compose

## Run command to get the code Backend, Frontend and Weights
```bash
chmod +x installer.sh
./installer.sh
```

## Edit your configurations
- CovidDocker/docker-compose.prod.yml
  ```bash
    backend:
      CLIENT_SECRET: {{YOUR_CLIENT_SECRET }}
      HOST_NAME: {{YOUR_HOSTNAME}}
    
    frontend:
      VUE_APP_BACKEND_URL: "http://{{YOUR_HOSTNAME}}:8000/"
    
    vision:
      PACS_IP: {{PACS_IP}} 
      PACS_PORT: {{PACS_PORT}}
      PACS_AE: GEPACS
      AE: {{PACKS_NAME}}
      DICOM_PORT: {{PACS_DICOM_PORT}}
    
    KEYCLOAK_HOSTNAME: {{YOUR_HOSTNAME}}
    KEYCLOAK_USER: {{KEYCLOAK_USERNAME}}
    KEYCLOAK_PASSWORD: {{KEYCLOAK_PASS}}
    
    
  ```
- CovidApp/client_secrets.json
  ```bash
  {
    "web": {
        "issuer": "http://{{YOUR_HOSTNAME}}:8080/auth/realms/{{YOUR_REALM_NAME}}",
        "auth_uri": "http://{{YOUR_HOSTNAME}}:8080/auth/realms/{{YOUR_REALM_NAME}}/protocol/openid-connect/auth",
        "client_id": "{{}YOUR_BACKEND_CLIENT_ID}",
        "client_secret": "{{SECRET_KEY_FROM_KEYCLOAK_CLIENT}}",
        "redirect_uris": [
            "http://{{YOUR_HOSTNAME}}:8000/oidc_callback"
        ],
        "userinfo_uri": "http://{{YOUR_HOSTNAME}}:8080/auth/realms/{{YOUR_REALM_NAME}}/protocol/openid-connect/userinfo",
        "token_uri": "http://{{YOUR_HOSTNAME}}:8080/auth/realms/{{YOUR_REALM_NAME}}/protocol/openid-connect/token",
        "token_introspection_uri": "http://{{YOUR_HOSTNAME}}:8080/auth/realms/{{YOUR_REALM_NAME}}/protocol/openid-connect/token/introspect",
        "bearer_only": "true"
      }
    }
    ```
 - CovidApp/blob/master/app.py
  ```bash
    'SECRET_KEY': '{{YOUR_SECRET_KEY}}',
    'OIDC_OPENID_REALM': '{{YOUR_REALM_NAME}}',
   ```
 - Covid19-frontend/.env.production
  ```bash
    VUE_APP_BACKEND_URL = "http://{{YOUR_HOSTNAME}}.com:8000"
    VUE_APP_KEYCLOAK_URL = "http://{{YOUR_HOSTNAME}}.com:8080"
    VUE_APP_KEYLOAK_REALM = "{{YOUR_ REALM_NAME}}"
    VUE_APP_KEYLOAK_CLIENT_ID = "{{YOUR_CLIENT_ID}"
  ```
- Covid19-frontend/blob/master/src/views/Result.vue
  ```bash
    var backend_url = "http://{{YOUR_HOSTNAME}}.com:8000/";
   ```
    
## Run command to run the application with Docker Compose
```bash
sudo docker-compose build
sudo docker-compose up
```

## Visit the url to test app
```http
http://{{HOSTNAME}}:{{YOUR_FRONTEND_PORT}} covidportal
http://{{HOSTNAME}}:8080 keycloak
http://{{HOSTNAME}}:8000 backend
http://{{HOSTNAME}}:1234 Databaseviewer
```
