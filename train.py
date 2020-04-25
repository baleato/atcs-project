import os
import time

from torch import load

from model import MetaLearner
from util import get_args, get_pytorch_device
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from transformers import BertTokenier

args = get_args()
device = get_pytorch_device(args)

# TODO: check that we can load learner with bert the same way
if args.resume_snapshot:
    model = load(args.resume_snapshot, map_location=device)
else:
    model = MetaLearner(config)
    model.to(device)

# TODO: Load datasets splits

writer = SummaryWriter(os.path.join(args.save_path, 'runs', '{}'.format(datetime.now())))

# TODO: training loop

writer.close()
