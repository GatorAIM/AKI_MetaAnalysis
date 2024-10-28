import pandas as pd

def read_config(site, suffix=None, config_base_location='/home/hoyinchan/code/AKI_CDM_PY/configs_files/publish_config/'):
    config = {}
    
    if suffix is None:
        suffix = site
    
    config['site'] = site
    config['config_base_location'] = config_base_location
    config['config_filename'] = f'configs_{suffix}.txt'
    
    filename = f"{config['config_base_location']}/{config['config_filename']}"
    
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace from the line and ignore everything after the '#' (comments)
            line = line.split('#', 1)[0].strip()
            # Ignore empty lines
            if not line:
                continue
            # Split the line at the first '=' to separate the key and value
            if '=' in line:
                key, value = line.split('=', 1)
                # Strip whitespace from the key and value
                key = key.strip()
                value = value.strip()
                # Optionally remove surrounding quotes from the value
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                # Store the key and value in the dictionary
                config[key] = value
    return config

def gen_config(site_list, config_base_location='/home/hoyinchan/code/AKI_CDM_PY/configs_files/publish_config/'):
    for site in site_list:
        # Read the config files for variables
        # The location of config files is at '/home/hoyinchan/code/AKI_CDM_PY/configs_files/publish_config/Configs_init.txt'
        configs_variables = utils_function.read_config(site, suffix='init', config_base_location='/home/hoyinchan/code/AKI_CDM_PY/configs_files/publish_config/')
        utils_function.write_config(configs_variables, suffix=site)

def write_config(config, suffix):
    
    config['config_filename'] = f"configs_{suffix}.txt"
    filename = f"{config['config_base_location']}/{config['config_filename']}"
    
    with open(filename, 'w') as file:
        for key, value in config.items():
            # Write the key and value pair as "key = value" in the file
            # If the value contains spaces or special characters, it can be enclosed in quotes
            if ' ' in value or any(c in value for c in '#;'):
                value = f'"{value}"'
            file.write(f'{key} = {value}\n')
            
def get_commons(config):
    
    site = config['site']    
    datafolder = config['datafolder']
    home_directory = config['home_directory']     
    
    return site, datafolder, home_directory

def get_bool_columns(configs_variables):
    dtype_collect = list()

    for configs_variable_m in  configs_variables:
        year=3000
        site_m, datafolder, home_directory = get_commons(configs_variable_m)

        datafolder = configs_variable_m['datafolder']
        stg = configs_variable_m['stg']
        fs = configs_variable_m['fs']
        oversample = configs_variable_m['oversample']
        model_type = configs_variable_m['model_type']   

        drop_correlation_catboost = configs_variable_m['drop_correlation_catboost']
        if drop_correlation_catboost:
            suffix = 'nc'
        else:
            suffix = ''    
#        X_test_m = pd.read_pickle(datafolder+site_m+'/X_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        X_test_m = pd.read_pickle(datafolder+site_m+'/bt3pos_'+site_m+'_'+stg+'_3000.pkl')
        dtype_collect.append(pd.DataFrame(X_test_m.dtypes).reset_index())
    
    dtype_collect = pd.concat(dtype_collect).drop_duplicates()
    dtype_collect.columns = ['index', 'dtype']
    dtype_collect[dtype_collect['dtype']==bool][['index']].to_parquet('/home/hoyinchan/code/AKI_CDM_PY/bool_columns.parquet')   