#!/bin/bash

# --- 1. Persiapan Variabel ---
export GOOGLE_CLOUD_PROJECT='taragbagas-465109'
export GOOGLE_CLOUD_REGION='us-central1'
export AR_REPO='gemini-repo'
export SERVICE_NAME='ta-bagaswahyu'

# PERBAIKAN: Gunakan format path lengkap, bukan hanya email
export DEPLOYER_SA_FULL_PATH="projects/$GOOGLE_CLOUD_PROJECT/serviceAccounts/gemini-deployer@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com"

export LOGS_BUCKET_URI="gs://${GOOGLE_CLOUD_PROJECT}_cloudbuild_logs/logs"

echo "============================================="
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Region: $GOOGLE_CLOUD_REGION"
echo "Service: $SERVICE_NAME"
echo "============================================="

# --- 2. Pastikan Artifact Registry Repository Ada ---
# Pesan "ALREADY_EXISTS" di sini tidak apa-apa dan bisa diabaikan.
gcloud artifacts repositories create "$AR_REPO" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --location="$GOOGLE_CLOUD_REGION" \
  --repository-format=Docker --quiet

# --- 3. Build & Submit Container Image dengan Format Service Account yang Benar ---
echo "Submitting build to Cloud Build..."
gcloud builds submit . \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --tag "$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$AR_REPO/$SERVICE_NAME:latest" \
  --service-account="$DEPLOYER_SA_FULL_PATH" \
  --gcs-log-dir="$LOGS_BUCKET_URI"

# --- 4. Deploy ke Cloud Run ---
echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --image="$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$AR_REPO/$SERVICE_NAME:latest" \
  --region=$GOOGLE_CLOUD_REGION \
  --platform=managed \
  --port=8080 \
  --memory=16Gi \
  --allow-unauthenticated \
  --update-secrets=HUGGING_FACE_HUB_TOKEN=hf-token:latest

echo "✅ Deployment successful!"