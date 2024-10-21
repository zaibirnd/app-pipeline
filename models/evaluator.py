import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils

from skimage.filters import threshold_otsu
import cv2
import copy
# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)
        self.model_str = args.net_G

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        #print(self.G_pred.shape)
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # if np.mod(self.batch_id, 1) == 1:
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        vis_pred = self._visualize_pred().detach().cpu().numpy()
        vis_pred = self.color_coding(vis_pred)
        print(vis_pred.shape)
        #thresh = threshold_otsu(vis_pred)
        #vis_pred = vis_pred > thresh
        vis_pred = utils.make_numpy_grid(torch.tensor(vis_pred))
        print(self.batch['L'].shape)
        vis_gt = self.color_coding(self.batch['L'])
        vis_gt = utils.make_numpy_grid(torch.tensor(vis_gt))
        #vis_pred = self.color_coding(vis_pred[:,:,0])
        #print(vis_pred.shape)
        vis = np.concatenate([vis_input, vis_input2, vis_gt ,vis_pred], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        file_name1 = os.path.join(
            self.vis_dir, 'eval_Res' + str(self.batch_id) + '.jpg')
        #vis_pred = np.clip(vis_pred, a_min=0.0, a_max=1.0)
        cv2.imwrite(file_name1, vis_pred)
        plt.imsave(file_name, vis)

    def color_coding(self, batch_input):
        #colors = [[0, 0, 255, 255], [0, 255, 0, 0], [255, 0, 0, 0]]
        colors = [[0, 255, 255], [0, 255, 0], [255, 0, 0]]
        batch_size = batch_input.shape[0]
        batch_output = []
        for batch_idx in range(batch_size):
            T = batch_input[batch_idx, :, :, :]
            #print(T)
            image_output = []
            for idx, pix in enumerate(colors):
                # print(pix)
                Tcopy = copy.deepcopy(T)
                T[T == 1] = pix[0]
                T[T == 2] = pix[1]
                T[T == 3] = pix[2]
                #T[T == 4] = pix[3]
                # print(T)
                image_output.append(T)
                T = Tcopy

            image_output = np.stack(image_output, axis=1).reshape(3, 1024, 1024)
            #print(TT)
            # TT_torch = torch.from_numpy(TT)
            # TT_torch = TT_torch.transpose((2,0,1))
            # TT = TT_torch.numpy()
            # cv2.imwrite(f'image_{i}.jpg', TT)
            # cv2.imshow('RGB', TT)
            # cv2.waitKey(1000)
            batch_output.append(image_output)
        batch_output = np.stack(batch_output, axis=0)
        #TTT_torch = torch.from_numpy(TTT)
        return batch_output

    def _collect_running_batch_states_new(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # if np.mod(self.batch_id, 1) == 1:
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        #print(self.G_pred.shape)
        vis_pred = self._visualize_pred().detach().cpu().numpy()
        #thresh = threshold_otsu(vis_pred)
        #vis_pred = vis_pred > thresh
        vis_pred = utils.make_numpy_grid(torch.tensor(vis_pred))

        vis_gt = utils.make_numpy_grid(self.batch['L'])
        vis = np.concatenate([vis_input, vis_input2,vis_pred ,  vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        file_name1 = os.path.join(
             self.vis_dir, 'eval_r' + str(self.batch_id) + '.jpg')
        plt.imsave(file_name, vis)
        plt.imsave(file_name1, vis_pred)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        # np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        
        if  self.model_str == "changeFormerV6":
            self.G_pred = self.net_G(img_in1, img_in2)[-1]
        else:
            self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self,checkpoint_name='best_ckpt.pt'):
        print('###############################',checkpoint_name)
        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        # self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                #print(batch['A'][:,:,100:110,100:110])
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
