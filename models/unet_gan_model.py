import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class UnetGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--constant-gp', type=float, default=100, help='constant of gradient')
            parser.add_argument('--lambda-gp', type=float, default=0.1, help='gradient penalty')

        return parser

    def __init__(self, opt, data_shape=(128, 128)):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.isTrain:
            self.loss_names = ['D_A', 'D2_A', 'G_A', 'G2_A', 'cycle_A', 'idt_A', 'D_B', 'D2_B', 'G_B', 'G2_B', 'cycle_B', 'idt_B' ] if opt.use_cycled_discriminators else ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'a10_b', 'real_B','fake_A', 'a10_a']
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D2_A', 'D2_B'] if opt.use_cycled_discriminators else ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.opt = opt

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.model_name, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        use_mask = opt.use_mask, raw_feat=opt.raw_feat, data_shape=data_shape,
                                        spectral_norm=opt.apply_spectral_norm)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.model_name, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        use_mask = opt.use_mask, raw_feat=opt.raw_feat, data_shape=data_shape,
                                        spectral_norm=opt.apply_spectral_norm)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            spectral_norm=opt.apply_spectral_norm)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            spectral_norm=opt.apply_spectral_norm)
            if opt.use_cycled_discriminators:
                self.netD2_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            spectral_norm=opt.apply_spectral_norm)
                self.netD2_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            spectral_norm=opt.apply_spectral_norm)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            if opt.use_cycled_discriminators:
                self.rec_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
                self.rec_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            disc_parameters = itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD2_A.parameters(), self.netD2_B.parameters()) if opt.use_cycled_discriminators else itertools.chain(self.netD_A.parameters(), self.netD_B.parameters())
            self.optimizer_D = torch.optim.Adam(disc_parameters, lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.use_mask:
            self.A_mask = input['A_mask' if AtoB else 'B_mask'].to(self.device)
            self.B_mask = input['B_mask' if AtoB else 'A_mask'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not self.opt.use_mask:
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        else:
            self.fake_B = self.netG_A(self.real_A, self.A_mask)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B, torch.ones(self.fake_B.size(0),1,self.fake_B.size(2),self.fake_B.size(3)))   # G_B(G_A(A))

            self.fake_A = self.netG_B(self.real_B, self.B_mask)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A , torch.ones(self.fake_A.size(0),1,self.fake_A.size(2),self.fake_A.size(3)))   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 + networks.cal_gradient_penalty(netD, real_data=real,
                                                            fake_data=fake.detach(), device=self.device,
                                                            constant=self.opt.constant_gp,
                                                            lambda_gp=self.opt.lambda_gp)[0]
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def backward_D2_A(self):
        """Calculate GAN loss for discriminator D2_A"""
        rec_A = self.rec_A_pool.query(self.rec_A)
        self.loss_D2_A = self.backward_D_basic(self.netD2_A, self.real_A, rec_A)

    def backward_D2_B(self):
        """Calculate GAN loss for discriminator D2_B"""
        rec_B = self.rec_B_pool.query(self.rec_B)
        self.loss_D2_B = self.backward_D_basic(self.netD2_B, self.real_B, rec_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            if self.opt.use_mask:
                self.idt_A  = self.netG_A(self.real_B, torch.ones(self.real_B.size(0),1,self.real_B.size(2),self.real_B.size(3)))
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B  = self.netG_B(self.real_A, torch.ones(self.real_A.size(0),1,self.real_A.size(2),self.real_A.size(3)))
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                self.idt_A  = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B  = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        if self.opt.use_cycled_discriminators:
            # Cycled Adversarial loss D2_A(G_B(G_A(A)))
            self.loss_G2_A = self.criterionGAN(self.netD2_A(self.rec_A), True)
            # Cycled Adversarial loss D2_B(G_A(G_B(B)))
            self.loss_G2_B = self.criterionGAN(self.netD2_B(self.rec_B), True)
        else:
            self.loss_G2_A = 0
            self.loss_G2_B = 0

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G2_A + self.loss_G2_B
        self.loss_G.backward()

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B

        if step % self.opt.G_update_frequency == 0:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights
        
        if step % self.opt.D_update_frequency == 0:
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            if self.opt.use_cycled_discriminators:
                self.backward_D2_A()      # calculate gradients for D2_A
                self.backward_D2_B()      # calculate graidents for D2_B
            self.optimizer_D.step()  # update D_A and D_B's weights

