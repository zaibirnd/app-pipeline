import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import json
from skimage.filters import threshold_otsu
import cv2
import copy
# Decide which device we want to run on
# torch.cuda.current_device()
from osgeo import gdal
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from skimage import io
import pandas as pd
from osgeo import gdal
from tqdm import tqdm

class CDEvaluator():

    def __init__(self, args, dataloader):
        self.args = args
        self.src = gdal.Open(glob.glob(args.input + '/*.tif')[0])
        self.geoTrans = self.src.GetGeoTransform()
        self.input = args.input

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        self.model_str = args.net_G

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        # logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        # self.logger = Logger(logger_path)
        # self.logger.write_dict_str(args.__dict__)

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
        self.checkpoint_dir = args.root + args.checkpoint_dir
        print(self.checkpoint_dir)
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            # self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            ck = os.path.join(self.checkpoint_dir, checkpoint_name)
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'], strict=False)

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            # self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
            #                   (self.best_val_acc, self.best_epoch_id))
            # self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _visualize_pred(self):
        # print(self.G_pred.shape)
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
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' % \
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # if np.mod(self.batch_id, 1) == 1:
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        vis_pred = self._visualize_pred().detach().cpu().numpy()
        vis_pred = self.color_coding(vis_pred)
        # thresh = threshold_otsu(vis_pred)
        # vis_pred = vis_pred > thresh
        vis_pred = utils.make_numpy_grid(torch.tensor(vis_pred))
        vis_gt = self.color_coding(self.batch['L'])
        vis_gt = utils.make_numpy_grid(torch.tensor(vis_gt))
        # vis_pred = self.color_coding(vis_pred[:,:,0])
        # print(vis_pred.shape)
        vis = np.concatenate([vis_input, vis_input2, vis_gt, vis_pred], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id) + '.jpg')
        file_name1 = os.path.join(
            self.vis_dir, 'eval_Res' + str(self.batch_id) + '.jpg')
        cv2.imwrite(file_name1, vis_pred)
        plt.imsave(file_name, vis)

    def color_coding(self, batch_input):
        colors = [[0, 0, 255, 255], [0, 255, 255, 0], [255, 0, 0, 0]]
        batch_size = batch_input.shape[0]
        batch_output = []
        for batch_idx in range(batch_size):
            T = batch_input[batch_idx, :, :, :]
            # print(T)
            image_output = []
            for idx, pix in enumerate(colors):
                # print(pix)
                Tcopy = copy.deepcopy(T)
                T[T == 1] = pix[0]
                T[T == 2] = pix[1]
                T[T == 3] = pix[2]
                T[T == 4] = pix[3]
                # print(T)
                image_output.append(T)
                T = Tcopy

            image_output = np.stack(image_output, axis=1).reshape(3, 1024, 1024)
            # print(TT)
            # TT_torch = torch.from_numpy(TT)
            # TT_torch = TT_torch.transpose((2,0,1))
            # TT = TT_torch.numpy()
            # cv2.imwrite(f'image_{i}.jpg', TT)
            # cv2.imshow('RGB', TT)
            # cv2.waitKey(1000)
            batch_output.append(image_output)
        batch_output = np.stack(batch_output, axis=0)
        # TTT_torch = torch.from_numpy(TTT)
        return batch_output

    def _collect_running_batch_states_new(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' % \
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # if np.mod(self.batch_id, 1) == 1:
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        # print(self.G_pred.shape)
        vis_pred = self._visualize_pred().detach().cpu().numpy()
        # thresh = threshold_otsu(vis_pred)
        # vis_pred = vis_pred > thresh
        vis_pred = utils.make_numpy_grid(torch.tensor(vis_pred))

        vis_gt = utils.make_numpy_grid(self.batch['L'])
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id) + '.jpg')
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

        if self.model_str == "changeFormerV6":
            self.G_pred = self.net_G(img_in1, img_in2)[-1]
        else:
            self.G_pred = self.net_G(img_in1, img_in2)

    def plot_polygons(self, polygons, mask):
        blank_image = np.zeros_like(mask)
        for polygon in polygons:
            cv2.polylines(blank_image, [polygon], isClosed=True, color=(255, 255, 255), thickness=1)
        return blank_image

    def get_polygon_color(self, polygon, mask):
        point  = polygon[0]
        x, y = point[0]
        return mask[y,x,:]

    def get_polygons(self, mask):
        df_local = pd.DataFrame(columns=['points'])
        df_global = pd.DataFrame(columns=['points_global'])
        # Ensure the image is binary
        _, binary_mask = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        # Find contours
        gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # contours, _ = cv2.findContours(binary_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(gray_image , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours to polygons
        polygons = [cv2.approxPolyDP(cnt, epsilon=0.01 * cv2.arcLength(cnt, True), closed=True) for cnt in contours]
        feature_arr = []
        for idx, polygon in tqdm(enumerate(polygons), total=len(polygons) ,desc='Generating Polygons'):
            # print('Length of polygon:  ',len(polygon))
            color = self.get_polygon_color(polygon, mask)
            polygon_global = self.pixel_to_world(self.geoTrans, [polygon])
            new_data_item = self.prepare_coords([polygon_global], color)
            feature_arr.append(new_data_item)
            arr_local = [polygon]
            arr_global = [polygon_global]
            df_local.loc[len(df_local.index)] = arr_local
            df_global.loc[len(df_global.index)] = arr_global
        self.create_class_file(feature_arr)
        return polygons, mask, df_local, df_global

    def create_class_file(self, feature_coords):
        # For this i would blame shazaib or appreciate for it
        features_dict = {'status': 404,
                         'msg': 'changes detected',
                         'result': {
                             "type": "FeatureCollection",
                             "features": feature_coords
                         }}
        with open(self.args.geojson + f'/json_file.geojson', 'w') as jf:
            json.dump(features_dict, jf)

    def prepare_coords(self, polygon,color):
        r,g,b = color
        r,g,b = int(r), int(g), int(b)

        return {
            "type": "feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": polygon
            },
            "properties": {
                "name": 'changes',
                "fill": "rgba'(255,255,255,1)",
                "fill-opacity": 0.4,
                "stroke": (r,g,b),
                "stroke-width": 2,
                "stroke-opacity": 1
            }
        }

    def pixel_to_world(self, geo_matrix, polygon):
        """
        Conversion of image pixels to crs co-ordinates
        """
        ul_x = geo_matrix[0]
        ul_y = geo_matrix[3]
        x_dist = geo_matrix[1]
        y_dist = geo_matrix[5]
        polygon_new = []
        for point in polygon[0]:
            x, y = point[0]
            corr_x = (ul_x + (x * x_dist))
            corr_y = (ul_y + (y * y_dist))
            # point_new = [[corr_x, corr_y, 0]]
            point_new = [corr_x, corr_y, 0]
            polygon_new.append(point_new)
        return polygon_new

    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        # self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        # self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        df_all = pd.DataFrame(columns=['points'])
        input_image = cv2.imread(glob.glob(self.input + '/*.tif')[0])
        blank_image = np.zeros_like(input_image)
        from tqdm import tqdm

        for self.batch_id, batch in tqdm(enumerate(self.dataloader, 0), total =len(self.dataloader),desc = 'Change Detection In Progress'):
            with torch.no_grad():
                vis_input = utils.make_numpy_grid(de_norm(batch['A']))
                vis_input2 = utils.make_numpy_grid(de_norm(batch['B']))
                name = batch['n'][0]
                self._forward_pass(batch)
                vis_pred = self._visualize_pred().detach().cpu().numpy()
                vis_pred = self.color_coding(vis_pred)
                output1 = utils.make_numpy_grid(torch.tensor(vis_pred))
                ref_x = name.split('_')[1]
                ref_y = name.split('_')[2]
                blank_image[int(ref_x):int(ref_x) + 1024, int(ref_y): int(ref_y) + 1024, :] = output1[:, :, :]

        polygons, mask, df_local, df_global = self.get_polygons(blank_image)
        poly_image = self.plot_polygons(polygons, mask)
        plt.imsave(self.args.geojson + '/poly_image.jpg', poly_image)
        plt.imsave(self.args.geojson + '/result.jpg', blank_image)
        df_local.to_csv(self.args.geojson + '/result_local.csv')
        df_global.to_csv(self.args.geojson + '/result_global.csv')
