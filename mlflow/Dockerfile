FROM ghcr.io/mlflow/mlflow:v2.13.0

# PostgreSQL (NeonDB backend store) + boto3 (S3 artifact store)
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

EXPOSE 5000

# Secrets à injecter dans le Space HF :
#   DATABASE_URL           postgresql://user:pass@host/mlflow_db?sslmode=require
#   ARTIFACT_ROOT          s3://cni-fraud-detection/getaround-mlflow/
#   AWS_ACCESS_KEY_ID      clé AWS
#   AWS_SECRET_ACCESS_KEY  clé secrète AWS
#   AWS_DEFAULT_REGION     eu-west-3

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${DATABASE_URL}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --serve-artifacts
