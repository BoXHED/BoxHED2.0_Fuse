import wandb
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--project-name', dest = 'project_name', help='project name for wandb')
parser.add_argument('--artifact-name', dest = 'artifact_name', help='artifact name for wandb')
parser.add_argument('--artifact-path', dest = 'artifact_path', help='csv path')
args = parser.parse_args()

df = pd.read_csv(args.artifact_path)
iris_table = wandb.Table(dataframe=df)

# Add the table to an Artifact to increase the row
# limit to 200000 and make it easier to reuse
iris_table_artifact = wandb.Artifact(args.artifact_name, type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

iris_table_artifact.add_file(args.artifact_path)

wandb.login(key=os.getenv('WANDB_KEY_TAMU'), relogin=True) # aa_ron_su
run = wandb.init(project=args.project_name)

# Log the table to visualize with a run...
run.log({args.artifact_name: iris_table})
run.log_artifact(iris_table_artifact)

print(f"Loaded artifact {args.artifact_name} from {args.artifact_path}")