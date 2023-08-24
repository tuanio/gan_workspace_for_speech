import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ClassificationModel(BaseModel):
    '''
        classification model for doing metrics between two domain
        1. train classification model on two domain
        
        Usage:
        - use last hidden state for clustering
        - calculate accuracy of input audio
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--model_name', type=str, default='conformer_classifier')
        return parser

    def __init__(self, opt, data_shape=(128, 128)):
        BaseModel.__init__(self, opt)
        if self.isTrain:
            self.loss_names = ['bce']
        if self.isTrain:
            self.model_names = ['G']

        self.opt = opt

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.model_name, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, use_mask = opt.use_mask,
                                        raw_feat=opt.raw_feat, data_shape=data_shape)

        if self.isTrain:
            self.criterion = torch.nn.BCEWithLogitsLoss()

            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.data = input['data'].to(self.device)
        self.label = input['label'].to(self.device)
        self.mask = input['mask'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.logits = self.netG(self.data * self.mask)  # G_A(A)

    def backward(self):
        # Forward cycle loss, bce with logits loss
        self.loss_bce = self.criterion(self.logits, self.label)
        self.loss_bce.backward()

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward()             # calculate gradients for G_A and G_B
        self.optimizer.step()       # update G_A and G_B's weights

