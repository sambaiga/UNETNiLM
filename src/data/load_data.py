import pandas as pd
import numpy as np

refit_appliance_data = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        'window':10,
        'on_power_threshold': 2000,
        'max_on_power': 3998
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        "window":50,
        'on_power_threshold': 50,
        'max_on_power': 3323
      
    },
    "dishwasher": {
        "mean": 700,
        "std": 1000,
        "window":100,
        'on_power_threshold': 10,
        'max_on_power': 3964
    },
    "washingmachine": {
        "mean": 400,
        "std": 700,
        "window":100,
        'on_power_threshold': 20,
        'max_on_power': 3999
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "window":10,
        'on_power_threshold': 200,
        'max_on_power': 3969,
    },
}

ukdale_appliance_data = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        'window':10,
        'on_power_threshold': 2000,
        'max_on_power': 3998
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        "window":50,
        'on_power_threshold': 50,
        
      
    },
    "dishwasher": {
        "mean": 700,
        "std": 700,
        "window":100,
        'on_power_threshold': 10
    },
    
    "washingmachine": {
        "mean": 400,
        "std": 700,
        "window":100,
        'on_power_threshold': 20,
        'max_on_power': 3999
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "window":10,
        'on_power_threshold': 200,
       
    },
}





aggregate_mean = 522
aggregate_std = 814

def binarization(data,threshold):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        threshold {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    state = np.where(data>= threshold,1,0).astype(int)
    return state
    
def get_percentile(data,p=50):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        quantile {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.percentile(data, p, axis=1, interpolation="nearest")

def generate_sequences(sequence_length, data):
    sequence_length = sequence_length - 1 if sequence_length% 2==0 else sequence_length
    units_to_pad = sequence_length // 2
    new_mains = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))
    new_mains = np.array([new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length+1)])
    return new_mains

def quantile_filter(sequence_length, data, p=50):
    new_mains = generate_sequences(sequence_length, data)
    new_mains = get_percentile(new_mains, p)
    return new_mains

def pre_process_data(data_type="test", data_path="../../data/REFIT/", save_path="../data/refit/"):
    
    data = []
    states = []
    #quantized_data = []
    aggregated = []
    size = []
    for app in list(refit_appliance_data.keys()):
        print(app)
        if data_type=="training":
            ram_size = 60429234
            power=pd.read_csv(f"{data_path}{app}/{app}_{data_type}_.csv", header=None).iloc[:ram_size,1].values.flatten()
            print(f"{len(power)}")
        else:    
            power=pd.read_csv(f"{data_path}{app}/{app}_{data_type}_.csv")[app].values.flatten()
        size.append(len(power))
        meter=quantile_filter(refit_appliance_data[app]['window'], power, p=50)
        #mu_data=mu_law(meter, max_value=appliance_data[app]['max_on_power'], mu=255, quantize=True)
        state = binarization(meter,refit_appliance_data[app]['on_power_threshold'])
        meter = (meter - refit_appliance_data[app]['mean'])/refit_appliance_data[app]['std']
        data.append(meter)
        states.append(state)
        aggregated.append(power)
        #quantized_data.append(mu_data)
    size = min(size)
    data = np.stack([d[:size] for d in data]).T
    states = np.stack([d[:size] for d in states]).T
    aggregated = np.stack([d[:size] for d in aggregated]).T.sum(1)
    aggregated = quantile_filter(10, aggregated, 50)
    aggregated = (aggregated - aggregate_mean)/aggregate_std
    
    del power, meter, state
    np.save(save_path+f"/{data_type}/inputs.npy", aggregated)
    np.save(save_path+f"/{data_type}/targets.npy", data)
    np.save(save_path+f"/{data_type}/states.npy", states)
    
def pre_process_uk_dale(data_type="training",  data_path="../../data/UKDALE/", save_path="../data/ukdale/"):
    targets = []
    states = [] 
    powers = []
    data = pd.read_csv(f"{data_path}ukdale_house_1_{data_type}.csv")
    columns = {'fridge freezer':'fridge', 'washer dryer':'washingmachine', 'dish washer':'dishwasher', 'kettle':'kettle', 'microwave':'microwave'}
    data.rename(columns, axis=1, inplace=True)

    for app in list(ukdale_appliance_data.keys()):
        power = data[app].values
        meter=quantile_filter(ukdale_appliance_data[app]['window'], power, p=50)
        state = binarization(meter,ukdale_appliance_data[app]['on_power_threshold'])
        meter = (meter - ukdale_appliance_data[app]['mean'])/ukdale_appliance_data[app]['std']
        power = (power - ukdale_appliance_data[app]['mean'])/ukdale_appliance_data[app]['std']
        targets.append(meter)
        states.append(state)
        powers.append(power)

    mains_denoise = data.sub_mains.values
    mains_denoise = quantile_filter(10, mains_denoise, 50)
    mains = data.mains.values-np.percentile(data.mains.values, 1)
    mains = np.where(mains <mains_denoise, mains_denoise, mains)
    mains = quantile_filter(10, mains, 50)
    mains_denoise = (mains_denoise - 123)/369
    mains = (mains - 389)/445
    states = np.stack(states).T
    targets = np.stack(targets).T
    powers = np.stack(powers).T
    del power, meter, state
    np.save(save_path+f"/{data_type}/denoise_inputs.npy", mains_denoise)
    np.save(save_path+f"/{data_type}/noise_inputs.npy", mains)
    np.save(save_path+f"/{data_type}/targets.npy", targets)
    np.save(save_path+f"/{data_type}/powers.npy", targets)
    np.save(save_path+f"/{data_type}/states.npy", states)    

if __name__ == "__main__":
    for data_type in ["test", "validation", "training"]:
        print(f"preprocess data for {data_type}")
        pre_process_uk_dale(data_type)