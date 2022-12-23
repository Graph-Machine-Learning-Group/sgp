import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config_file = os.path.join(base_dir, 'config.yaml')
config = {
    'config_dir': os.path.join(base_dir, 'config'),
    'data_dir': os.path.join(base_dir, 'datasets'),
    'logs_dir': os.path.join(base_dir, 'log')
}
