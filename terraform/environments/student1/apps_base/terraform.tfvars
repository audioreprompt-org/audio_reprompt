region                         = "us-east-1"
owner                          = "4AI"

db_publicly_accessible   = true
sg_ingress_cidr_blocks   = ["0.0.0.0/0"]
secret_name              = "audio-db-secrets"
db_allocated_storage_gib = 64
db_name                  = "audioreprompt"
db_username              = "audiouser"
db_password              = "passAudio25"
