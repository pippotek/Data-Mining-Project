import yaml

def load_config(file_path):
    """
    Loads a YAML configuration file and returns the contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary with configuration values.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
