"""Helper function to handle extraction of configurations

"""
  
import configparser

# edit config file path accordingly 
BASE_CONFIG_PATH = '../config/config.ini'


def configfile(base_config_path=BASE_CONFIG_PATH):
    """
    Read configurations from base config file and return a ConfigParser object

    Example usage:
    configfile = configfile()
    database_url = configfile.get('database', 'database_url')  # to get database_url parameter from database section

    Parameters:
    base_config_path (str): Directory path to config file

    Returns:
    config: ConfigParser object that can get config parameters 
    """
    config=configparser.ConfigParser()
    config.read(base_config_path)
    return config


if __name__=='__main__':
    configfile = configfile()
    database_url = configfile.get('database', 'database_url')
    print(database_url)