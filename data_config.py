
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'xBDataset':
            self.root_dir = 'data/xbd/'
        elif data_name == 'TestInput':
            self.root_dir = os.path.expanduser('~') + '/.local/share/QGIS/QGIS3/profiles/default/python/plugins/atr/CD_pipeline_multi/data/construction/'
        elif data_name == 'quick_start':
            self.root_dir = 'samples'
        elif data_name == 'LEVIR':
            self.root_dir = 'data/LEVIR_CD/'
        elif data_name == 'my_test':
            self.root_dir = 'samples/my_test/'
            
        elif data_name == 'construction_c':
            self.root_dir = 'data/construction/'
        
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

