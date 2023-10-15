import wandb
import pandas as pd
import argparse
import os

def drop_ts(df : pd.DataFrame):
    ''' drops all unnecessary timeseries columns
    '''

    to_drop = [
        'Capillary refill rate',
        'Diastolic blood pressure',
        'Fraction inspired oxygen',
        'Glascow coma scale total',
        'Glucose',
        'Heart Rate',
        'Height',
        'Mean blood pressure',
        'Oxygen saturation',
        'Respiratory rate',
        'Systolic blood pressure',
        'Temperature',
        'Weight',
        'pH',
        'Ethnicity',
        'Gender',
        'Age',
        'Height_static',
        'Weight_static',
        'Inspired O2 Fraction',
        'Respiratory Rate',
        'O2 saturation pulseoxymetry',
        'PEEP set',
        'Inspired Gas Temp.',
        'Paw High',
        'Vti High',
        'Fspn High',
        'Apnea Interval',
        'Tidal Volume (set)',
        'Tidal Volume (observed)',
        'Minute Volume',
        'Respiratory Rate (Set)',
        'Respiratory Rate (spontaneous)',
        'Respiratory Rate (Total)',
        'Peak Insp. Pressure',
        'Plateau Pressure',
        'Mean Airway Pressure',
        'Total PEEP Level',
        'Inspiratory Time',
        'Expiratory Ratio',
        'Inspiratory Ratio',
        'Ventilator Tank #1',
        'Ventilator Tank #2',
        'Tidal Volume (spontaneous)',
        'PSV Level',
        'O2 Flow',
        'O2 Flow (additional cannula)',
        'Flow Rate (L/min)',
        'RCexp (Measured Time Constant)',
        'Compliance',
        'Pminimum',
        'Pinsp (Hamilton)',
        'Resistance Insp',
        'Resistance Exp',
        'CO2 production',
        'Cuff Pressure',
        'ETT Position Change',
        'ETT Re-taped',
        'Spont Vt',
        'Spont RR',
        'MDI #1 Puff',
        'Cuff Volume (mL)',
        'In-line Suction Catheter Changed',
        'Trach Care',
        'MDI #2 Puff',
        'MDI #3 Puff',
        'Negative Insp. Force',
        'Vital Cap',
        'BiPap O2 Flow',
        'PCV Level',
        'BiPap EPAP',
        'BiPap IPAP',
        'Pinsp (Draeger only)',
        'Recruitment Duration',
        'PeCO2',
        'Recruitment Press',
        'Nitric Oxide',
        'Nitric Oxide Tank Pressure',
        'Transpulmonary Pressure (Exp. Hold)',
        'Transpulmonary Pressure (Insp. Hold)',
        'P High (APRV)',
        'P Low (APRV)',
        'T High (APRV)',
        'T Low (APRV)',
        'Small Volume Neb Dose #2',
        '% Minute Volume',
        'ATC %',
        'BiPap bpm (S/T -Back up)',
        'Peak Exp Flow Rate',
        'Vd/Vt Ratio',
        'Resting Energy Expenditure',
        'Resistance',
        'O2 Consumption',
        'Respiratory Quotient',
        '#past_IVs',
        't_from_last_IV_t_start',
        't_from_last_IV_t_end']
    
    df.drop(to_drop, axis = 1, inplace=True)
    return df

def log_artifact(artifact_path, artifact_name, project_name, artifact_description = None, artifact_metadata = None, verbose = True, do_filter = False):
    df = pd.read_csv(artifact_path)
    if do_filter:
        drop_ts(df)
    table = wandb.Table(dataframe=df)

    # Add the table to an Artifact to increase the row
    # limit to 200000 and make it easier to reuse
    artifact = wandb.Artifact(artifact_name,
                            type = "dataset",
                            description = artifact_description,
                            metadata = artifact_metadata,)
    artifact.add(table, "iris_table")

    artifact.add_file(artifact_path)

    # wandb.login(key=os.getenv('WANDB_KEY_TAMU'), relogin=True) # aa_ron_su
    wandb.login(key=os.getenv('WANDB_KEY_PERSONAL'), relogin=True) # aa_ron_su
    run = wandb.init(project=project_name)

    # Log the table to visualize with a run...
    run.log({artifact_name: table})
    run.log_artifact(artifact)

    if verbose:
        print(f"Loaded artifact {artifact_name} from {artifact_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-name', dest = 'project_name', help='project name for wandb')
    parser.add_argument('--artifact-name', dest = 'artifact_name', help='artifact name for wandb')
    parser.add_argument('--artifact-path', dest = 'artifact_path', help='csv path')
    args = parser.parse_args()

    log_artifact(**dict(args._get_kwargs()))