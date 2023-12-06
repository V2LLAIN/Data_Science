import wandb
from util import get_config_dict
from config import args

notes = ""

if args.WANDB:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb.login(key='eed81e1c0a41dd8dd67a4ca90cea1be5a06d4eb0')
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    run = wandb.init(project=args.wandb_project,
                     name=args.wandb_name,
                     entity=args.wandb_entity,
                     config=get_config_dict(args),
                     job_type="train",
                     notes=notes,
                     anonymous=anony)