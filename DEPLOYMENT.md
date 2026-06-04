# Deployment Guide

This project is ready for Docker-based deployment to AWS Elastic Beanstalk and Azure App Service.

## Before deploying

The application needs trained artifacts:

- `artifacts/model.pkl`
- `artifacts/preprocessor.pkl`

The Docker image creates these automatically during build by running:

```bash
python -m src.components.data_ingestion
```

## Run locally with Docker

```bash
docker build -t student-performance .
docker run -p 5000:5000 student-performance
```

Open:

```text
http://127.0.0.1:5000
```

## Deploy to AWS Elastic Beanstalk

Install and configure the EB CLI first:

```bash
pip install awsebcli
aws configure
```

Create and deploy the Elastic Beanstalk app:

```bash
eb init student-performance --platform docker --region ap-south-1
eb create student-performance-env
eb deploy
eb open
```

Elastic Beanstalk will detect the `Dockerfile`, build the image, train the model during the image build, and run the Flask app with Gunicorn.

## Deploy to Azure App Service

Install and sign in with the Azure CLI:

```bash
az login
```

Create the Azure resources:

```bash
az group create --name student-performance-rg --location centralindia
az appservice plan create --name student-performance-plan --resource-group student-performance-rg --sku B1 --is-linux
```

Build and push the Docker image to Azure Container Registry:

```bash
az acr create --resource-group student-performance-rg --name studentperformanceacr --sku Basic
az acr login --name studentperformanceacr
docker build -t studentperformanceacr.azurecr.io/student-performance:latest .
docker push studentperformanceacr.azurecr.io/student-performance:latest
```

Create the Azure web app from that container image:

```bash
az webapp create --resource-group student-performance-rg --plan student-performance-plan --name student-performance-app-unique --deployment-container-image-name studentperformanceacr.azurecr.io/student-performance:latest
az webapp config container set --name student-performance-app-unique --resource-group student-performance-rg --docker-custom-image-name studentperformanceacr.azurecr.io/student-performance:latest
az webapp config appsettings set --name student-performance-app-unique --resource-group student-performance-rg --settings WEBSITES_PORT=5000
az webapp browse --name student-performance-app-unique --resource-group student-performance-rg
```

Replace `student-performance-app-unique` and `studentperformanceacr` with globally unique names.
