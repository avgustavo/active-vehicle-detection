from os import linesep
from pathlib import Path
from datetime import datetime
import platform 
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType, DatasourcePurpose
import torch


###### CHANGE THESE 2 VARIABLES
LIGHTLY_TOKEN = "6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8"  # Copy from https://app.lightly.ai/preferences
DATASET_PATH = Path("clothing_dataset")  # e.g., Path("/path/to/images") or Path("clothing_dataset")
######

assert DATASET_PATH.exists(), f"Dataset path {DATASET_PATH} does not exist."

# Create the LightlyOne client to connect to the API.
client = ApiWorkflowClient(token=LIGHTLY_TOKEN)

# Create the dataset on the LightlyOne Platform.
# See our guide for more details and options:
# https://docs.lightly.ai/docs/set-up-your-first-dataset
client.create_dataset(
    dataset_name=f"first_dataset__{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}",
    dataset_type=DatasetType.IMAGES,
)

# Configure the datasources.
# See our guide for more details and options:
# https://docs.lightly.ai/docs/set-up-your-first-dataset
client.set_local_config(purpose=DatasourcePurpose.INPUT)
client.set_local_config(purpose=DatasourcePurpose.LIGHTLY)

# Schedule a run on the dataset to select 50 diverse samples.
# See our guide for more details and options:
# https://docs.lightly.ai/docs/run-your-first-selection
scheduled_run_id = client.schedule_compute_worker_run(
    worker_config={"shutdown_when_job_finished": True},
    selection_config={
        "n_samples": 10,
        "strategies": [
            {"input": {"type": "EMBEDDINGS"}, "strategy": {"type": "DIVERSITY"}}
        ],
    },
)

for run_info in client.compute_worker_run_info_generator(
    scheduled_run_id=scheduled_run_id
):
    print(
        f"LightlyOne Worker run is now in state='{run_info.state}' with message='{run_info.message}'"
    )

if run_info.ended_successfully():
    print("SUCCESS")
else:
    print("FAILURE")

# Print the next commands
gpus_flag = "--gpus all" if torch.cuda.is_available() else ""
omp_num_threads_flag = " -e OMP_NUM_THREADS=1" if platform.system() == "Darwin" else ""
print(
    f"{linesep}Docker Run command: {linesep}"
    f"\033[7m"
    f"docker run{gpus_flag}{omp_num_threads_flag} --shm-size='1024m' --rm -it \\{linesep}"
    f"\t-v '{DATASET_PATH.absolute()}':/input_mount:ro \\{linesep}"
    f"\t-v '{Path('lightly').absolute()}':/lightly_mount \\{linesep}"
    f"\t-e LIGHTLY_TOKEN={LIGHTLY_TOKEN} \\{linesep}"
    f"\tlightly/worker:latest{linesep}"
    f"\033[0m"
)
print(
    f"{linesep}Lightly Serve command:{linesep}"
    f"\033[7m"
    f"lightly-serve \\{linesep}"
    f"\tinput_mount='{DATASET_PATH.absolute()}' \\{linesep}"
    f"\tlightly_mount='{Path('lightly').absolute()}'{linesep}"
    f"\033[0m"
)
