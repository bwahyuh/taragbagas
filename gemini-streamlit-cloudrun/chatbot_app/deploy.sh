#!/bin/bash

# ==============================================================================
# CATATAN PENTING TENTANG IZIN (PERMISSIONS)
# ==============================================================================
# Skrip ini melakukan tindakan administratif di Google Cloud Project Anda,
# seperti membuat Service Account dan memberikan izin IAM.
#
# !! KESALAHAN UMUM & SOLUSINYA !!
# Jika Anda melihat error "PERMISSION_DENIED" atau "does not have permission... setIamPolicy",
# itu berarti akun yang Anda gunakan untuk menjalankan skrip ini (misalnya,
# dari Cloud Shell atau VM) tidak memiliki wewenang yang cukup.
#
# UNTUK MENGATASINYA: Sebelum menjalankan skrip ini, pastikan Anda telah login
# dengan akun yang memiliki peran 'Owner'. Jalankan perintah berikut di terminal:
#
#   1. gcloud auth login
#      (Pilih akun email Anda yang merupakan Owner proyek)
#
#   2. gcloud config set project [ID_PROYEK_ANDA]
#
# Jika Anda *tetap* mengalami error, berarti akun yang menjalankan skrip
# (misalnya service account VM) perlu diberi izin oleh seorang Owner.
# Owner proyek harus menjalankan perintah seperti ini:
#
#   gcloud projects add-iam-policy-binding [ID_PROYEK_ANDA] \
#       --member="serviceAccount:[SERVICE_ACCOUNT_EMAIL_PELAKU]" \
#       --role="roles/resourcemanager.projectIamAdmin"
#
#   gcloud projects add-iam-policy-binding [ID_PROYEK_ANDA] \
#       --member="serviceAccount:[SERVICE_ACCOUNT_EMAIL_PELAKU]" \
#       --role="roles/iam.serviceAccountAdmin"
#
# ==============================================================================


# --- 1. Persiapan Variabel ---
set -e # Hentikan skrip jika ada perintah yang gagal

export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)
export GOOGLE_CLOUD_REGION='us-central1'
export AR_REPO='gemini-repo'
export SERVICE_NAME='ta-bagaswahyu'
export DEPLOYER_SA="gemini-deployer"
export DEPLOYER_SA_EMAIL="${DEPLOYER_SA}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"
export DEPLOYER_SA_FULL_PATH="projects/${GOOGLE_CLOUD_PROJECT}/serviceAccounts/${DEPLOYER_SA_EMAIL}"
export PROJECT_NUMBER=$(gcloud projects describe $GOOGLE_CLOUD_PROJECT --format="value(projectNumber)")
export CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"


echo "============================================="
echo "Project ID:      $GOOGLE_CLOUD_PROJECT"
echo "Service:         $SERVICE_NAME"
echo "Deployer SA:     $DEPLOYER_SA_EMAIL"
echo "Build SA Path:   $DEPLOYER_SA_FULL_PATH"
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
if ! gcloud iam service-accounts describe "$DEPLOYER_SA_EMAIL" --project="$GOOGLE_CLOUD_PROJECT" > /dev/null 2>&1; then
    echo "Membuat Service Account: $DEPLOYER_SA..."
    gcloud iam service-accounts create "$DEPLOYER_SA" \
        --display-name="Gemini Deployer SA" \
        --project="$GOOGLE_CLOUD_PROJECT"
fi

# --- 4. Berikan Izin yang Diperlukan ---
echo "🔄 Memberikan izin IAM untuk Deployer SA (mungkin tidak ada output jika izin sudah ada)..."
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/run.admin" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/iam.serviceAccountUser" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/artifactregistry.writer" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/storage.admin" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/secretmanager.secretAccessor" --condition=None --quiet
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member="serviceAccount:$DEPLOYER_SA_EMAIL" --role="roles/aiplatform.user" --condition=None --quiet

# Izin untuk Cloud Build SA: bisa 'meniru' Deployer SA selama proses build
echo "🔄 Memberikan izin untuk Cloud Build SA..."
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA_EMAIL" --member="serviceAccount:$CLOUDBUILD_SA" --role="roles/iam.serviceAccountTokenCreator" --quiet


# --- 5. Pastikan Artifact Registry Repository Ada ---
echo "🔄 Memeriksa/Membuat Artifact Registry Repo..."
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$GOOGLE_CLOUD_REGION" --project="$GOOGLE_CLOUD_PROJECT" > /dev/null 2>&1; then
    echo "Membuat Artifact Registry Repo: $AR_REPO..."
    gcloud artifacts repositories create "$AR_REPO" \
      --project="$GOOGLE_CLOUD_PROJECT" \
      --location="$GOOGLE_CLOUD_REGION" \
      --repository-format=Docker --quiet
fi

# --- 5b. Pastikan GCS Bucket untuk Logs Ada ---
LOGS_BUCKET="gs://${GOOGLE_CLOUD_PROJECT}_cloudbuild"
echo "🔄 Memeriksa/Membuat GCS Bucket untuk Logs di $LOGS_BUCKET..."
if ! gcloud storage buckets describe "$LOGS_BUCKET" --project="$GOOGLE_CLOUD_PROJECT" > /dev/null 2>&1; then
    echo "Membuat GCS Bucket: $LOGS_BUCKET..."
    gcloud storage buckets create "$LOGS_BUCKET" --project="$GOOGLE_CLOUD_PROJECT" --location="$GOOGLE_CLOUD_REGION"
fi


# --- 6. Build & Submit Container Image ---
echo "🏗️ Memulai proses build dengan Cloud Build..."
gcloud builds submit . \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --tag "$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$AR_REPO/$SERVICE_NAME:latest" \
  --service-account="$DEPLOYER_SA_FULL_PATH" \
  --gcs-log-dir="${LOGS_BUCKET}/logs"

echo "✅ Build selesai."

# --- 7. Deploy ke Cloud Run ---
echo "🚀 Mendeploy ke Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --image="$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$AR_REPO/$SERVICE_NAME:latest" \
  --region=$GOOGLE_CLOUD_REGION \
  --platform=managed \
  --port=8080 \
  --memory=16Gi \
  --allow-unauthenticated \
  --service-account="$DEPLOYER_SA_EMAIL" \
  --update-secrets=HUGGING_FACE_HUB_TOKEN=hf-token:latest

echo "✅ Deployment selesai!"

