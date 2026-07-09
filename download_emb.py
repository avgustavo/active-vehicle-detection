from pathlib import Path
from lightly.api import ApiWorkflowClient

from pipeline import configure_lightly_client, monitoring_run

def retrieve_embeddings():
    """
    Retrieve embeddings from the Lightly API and save them to a CSV file.
    """
    # client = configure_lightly_client("6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8", "d10_all")

    # scheduled_run_id = client.schedule_compute_worker_run(
    #     worker_config = {
    #         "shutdown_when_job_finished": True,
    #         "use_datapool": True,
    #         "datasource": {
    #             "process_all": True,
    #         },
    #         "enable_training": True,
    #     },
    #     selection_config={
    #         "proportion_samples": 1, # 1% do dataset
    #         "strategies": [
    #             {
    #                 "input": {
    #                     "type": "RANDOM",
    #                     "random_seed": 42, # optional, for reproducibility
    #                 },
    #                 "strategy": {
    #                     "type": "WEIGHTS",
    #                 }
    #             }
    #         ]
    #     },
    # )
    # monitoring_run(client, scheduled_run_id)
    # print_commands()


    # Ensure the directory exists
    Path('lightly/.lightly/embeddings').mkdir(parents=True, exist_ok=True)

    # Create the LightlyOne client to connect to the API.
    client = ApiWorkflowClient(token="6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8")

    client.set_dataset_id_by_name("d10_all")

    client.download_embeddings_csv(output_path="lightly/.lightly/embeddings/d10_embedding.csv")


retrieve_embeddings()