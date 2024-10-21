import concurrent.futures
import os
import cv2
import shutil

class CroppingThread:
    def __init__(self,args):
        self.input = args.input
        self.step = args.step_size
        self.patch_size = args.patch_size
        self.step_size = self.patch_size - int(self.patch_size * self.step)
        # self.output_path = 'data/construction/test_cases/images/'
        self.output_path = args.root + 'data/construction/test_cases/images/'
        self.exists_rm_create(self.output_path)
        self.root_path = os.getcwd()

    def slicethread(self, image_path):
        points = self.divisions_new(image_path)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.cropping, points)
            for result in results:
                pass

    def exists_rm_create(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.mkdir(path)

    def split_name_from_path(self, path):
        file_name = path.split('/')[-1]
        return file_name

    def divisions_old(self, image_path):
        self.image_array = {}
        name_ext = self.split_name_from_path(image_path)
        name = name_ext.split('.')[0]
        image = cv2.imread(image_path)
        self.image_array[name] = image
        (height, width, channels) = image.shape
        w1 = self.step_size
        h1 = self.step_size

        points = []
        for y in range(0, height-h1 , h1):
            for x in range(0, width-w1 , w1):
                start_x = x
                start_y = y
                point = (start_x, start_y, name)
                points.append(point)
                # break
        return points

    def divisions_new(self,image_path):
        self.image_array = {}
        name_ext = self.split_name_from_path(image_path)
        name = name_ext.split('.')[0]
        image = cv2.imread(image_path)
        self.image_array[name] = image
        (height, width, channels) = image.shape
        w1 = self.step_size
        h1 = self.step_size

        points = []
        for y in range(0, height-(height%h1) + 1 , h1):
            for x in range(0, width-(width%w1)+ 1 , w1):

                if x + self.patch_size > width:
                    diff = width - x
                    add_value = self.patch_size - diff
                    x = x - add_value
                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)

                elif y + self.patch_size > height:
                    diff = height - y
                    add_value = self.patch_size - diff
                    y = y - add_value
                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)

                elif (x + self.patch_size > width) and (y + self.patch_size > height):
                    diff = width - x
                    add_value = self.patch_size - diff
                    x = x - add_value

                    diff = height - y
                    add_value = self.patch_size - diff
                    y = y - add_value

                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)

                else:
                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)
                    # break
        return points

    def plot_crops(self,x1,y1,w,h, name):

        x2 = x1 + w
        y2 = y1 + h

        cv2.rectangle(self.image_array[name], (x1,y1), (x2,y2), (0,0,255), 3) 

    def cropping(self, point):
        (x1, y1, name) = point
        w = self.patch_size
        h = self.patch_size
        image = self.image_array[name]

        image_patch = self.image_array[name][y1:y1+h, x1:x1+w]
        #cv2.imwrite(self.output_path + name + '_' + str(y1) + '_' + str(x1) + '_pre' + '.jpg', image_patch)
        n = name
        end_name = n.split('_')[-1]
        name = name.replace( '_' + end_name, '')
        if end_name=='pre':
            name_new = name.replace('_pre', '')
            cv2.imwrite(self.output_path + name + '_' + str(y1) + '_' + str(x1) + '_pre' +'.png', image_patch)
        else:
            name_new = name.replace('_post', '')
            cv2.imwrite(self.output_path + name + '_' + str(y1) + '_' + str(x1) + '_post' + '.png', image_patch)






