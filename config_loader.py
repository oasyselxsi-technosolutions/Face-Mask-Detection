import configparser

class ConfigLoader:
    def __init__(self, config_path="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_email_config(self):
        return self.config['email']

    def get_snapshot_interval(self):
        return int(self.config['snapshot']['interval_minutes'])