#!/bin/bash

# --- 1. Persiapan Variabel ---
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)
export GOOGLE_CLOUD_REGION='us-central1'
export AR_REPO='gemini-repo'
export SERVICE_NAME='ta-bagaswahyu'
export DEPLOYER_SA="gemini-deployer"
export DEPLOYER_SA_EMAIL="${DEPLOYER_SA}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"
export DEPLOYER_SA_FULL_PATH="projects/${GOOGLE_CLOUD_PROJECT}/serviceAccounts/${DEPLOYER_SA_EMAIL}"
export PROJECT_NUMBER=$(gcloud projects describe $GOOGLE_CLOUD_PROJECT --format="value(projectNumber)")
export CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
export LOGS_BUCKET_URI="gs://${GOOGLE_CLOUD_PROJECT}_cloudbuild/logs"


echo "============================================="
echo "Project ID:      $GOOGLE_CLOUD_PROJECT"
echo "Service:         $SERVICE_NAME"
echo "Deployer SA:     $DEPLOYER_SA_EMAIL"
echo "Build SA Path:   $DEPLOYER_SA_FULL_PATH"
echo "Logs Bucket:     $LOGS_BUCKET_URI"
echo "============================================="


# --- 2. Aktifkan API yang Diperlukan ---
echo "🔄 Mengaktifkan API yang diperlukan..."
gcloud services enable iam.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    aiplatform.googleapis.com --project="$GOOGLE_CLOUD_PROJECT" --quiet

# --- 3. Buat Service Account untuk Deploy (jika belum ada) ---
echo "🔄 Memeriksa/Membuat Service Account Deployer..."
gcloud iam service-accounts describe "$DEPLOYER_SA_EMAIL" --project="$GOOGLE_CLOUD_PROJECT" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Membuat Service Account: $DEPLOYER_SA..."
    gcloud iam service-accounts create "$DEPLOYER_SA" \
        --display-name="Gemini Deployer SA" \
        --project="$GOOGLE_CLOUD_PROJECT"
fi

# --- 4. Berikan Izin yang Diperlukan ---
echo "🔄 Memberikan izin IAM (mungkin tidak ada output jika izin sudah ada)..."
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/run.admin" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/iam.serviceAccountUser" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/artifactregistry.writer" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/storage.admin" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/secretmanager.secretAccessor" --condition=None --quiet
# --- SOLUSI: Tambahkan izin untuk menggunakan model Vertex AI ---
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/aiplatform.user" --condition=None --quiet

# Izin untuk Cloud Build SA: bisa 'meniru' Deployer SA selama proses build
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA_EMAIL" --member="serviceAccount:$CLOUDBUILD_SA" --role="roles/iam.serviceAccountTokenCreator" --quiet


# --- 5. Pastikan Artifact Registry Repository Ada ---
echo "🔄 Memeriksa/Membuat Artifact Registry Repo..."
gcloud artifacts repositories describe "$AR_REPO" --location="$GOOGLE_CLOUD_REGION" --project="$GOOGLE_CLOUD_PROJECT" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Membuat Artifact Registry Repo: $AR_REPO..."
    gcloud artifacts repositories create "$AR_REPO" \
      --project="$GOOGLE_CLOUD_PROJECT" \
      --location="$GOOGLE_CLOUD_REGION" \
      --repository-format=Docker --quiet
fi


# --- 6. Build & Submit Container Image ---
echo "🏗️ Memulai proses build dengan Cloud Build..."
gcloud builds submit . \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --tag "$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$AR_REPO/$SERVICE_NAME:latest" \
  --service-account="$DEPLOYER_SA_FULL_PATH" \
  --gcs-log-dir="$LOGS_BUCKET_URI"

# Jika build gagal, hentikan skrip
if [ $? -ne 0 ]; then
    echo "❌ Build Gagal. Proses deployment dihentikan."
    exit 1
fi

# --- 7. Deploy ke Cloud Run ---
echo "🚀 Mendeploy ke Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --image="$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$AR_REPO/$SERVICE_NAME:latest" \
  --region=$GOOGLE_CLOUD_REGION \
  --platform=managed \
  --port=8080 \
  --memory=4Gi \
  --allow-unauthenticated \
  --service-account="$DEPLOYER_SA_EMAIL" \
  --update-secrets=HUGGING_FACE_HUB_TOKEN=hf-token:latest

echo "✅ Deployment selesai!"
