from pathlib import Path
from lightly.api import ApiWorkflowClient

Path('lightly/.lightly/embeddings').mkdir(parents=True, exist_ok=True)

# Create the LightlyOne client to connect to the API.
client = ApiWorkflowClient(token="6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8")

client.set_dataset_id_by_name("testssl")

client.download_embeddings_csv(output_path="lightly/.lightly/embeddings/transitar_embedding.csv")