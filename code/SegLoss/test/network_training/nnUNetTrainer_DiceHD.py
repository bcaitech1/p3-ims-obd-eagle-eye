from nnunet.training.loss_functions.boundary_loss import DC_and_HD_loss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_DiceHD(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.loss = DC_and_HD_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False},{})
