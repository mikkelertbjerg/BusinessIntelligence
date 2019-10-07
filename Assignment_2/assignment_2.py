import pandas as pd
import glob, os, zipfile, shutil
from os import listdir
from osmread import parse_file, Node
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

def haversine_distance(origin, destination):

    lat_orig, lon_orig = origin
    lat_dest, lon_dest = destination
    radius = 6371

    dlat = math.radians(lat_dest-lat_orig)
    dlon = math.radians(lon_dest-lon_orig)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat_orig)) 
        * math.cos(math.radians(lat_dest)) * math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def distance_arr():
    dest_coords = [55.65, 12.083333]
    distance_arr = []
    for idx, decoded_node in enumerate(decode_node()):
        distance.concat(haversine_distance(decoded_node, dest_coords))
    return distance_arr

def decode_node():
    for entry in parse_file('./denmark-latest.osm'):
        if (isinstance(entry, Node) and 
            'lat' in entry and 
            'lon' in entry):
            yield [entry.lat, entry.lon]

def gen_scatter():
    data = np.array([])

    for idx, decoded_node in enumerate(decode_node()):
        data.concat(decoded_node)
    x, y = data.T
    plt.scatter(x,y)
    plt.title("Danish Housing Coordinates")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def zip_df_csv(dfs, zip):
    df_norm = pd.DataFrame(columns=['zip', 'pp_sq_m'])
    for df in dfs:
        newdf = df[df.zip == zip]
        df_row = pd.concat([df_norm, newdf])
    csv = df_norm.to_csv (r'C:\Users\seana\Desktop\Sem6\BI\BI\Assignment2\zip'+zip+'.csv', index = None, header=True)
    return csv

def decode_node_to_csv():
    mycount = 0
    postcodes = defaultdict(list)
    for entry in parse_file('./denmark-latest.osm'):
        mycount += 1
        if (isinstance(entry, Node) and 
            'addr:street' in entry.tags and 
            'addr:postcode' in entry.tags and 
            'addr:city' in entry.tags and
            'lat' in entry and
            'lon' in entry and
            'addr:housenumber' in entry.tags):
                address_field = entry.tags['addr:city']
                zip_field = entry.tags['addr:postcode']
                search_address = ' '.join([zip_field,address_field])
                street_field = entry.tags['addr:street']
                houseno_field = entry.tags['addr:housenumber']
                search_street = ' '.join([street_field,houseno_field])
                my_tuple = (search_street, entry.lat, entry.lon)
                postcodes[search_address].append(my_tuple)
        print(mycount)
    return postcodes
            #yield entry

def calc_price_averages(df, zipcodes, inpyear):
    values_dict = defaultdict(list)
    ret_df = pd.DataFrame(columns=['zip', 'pp_sq_m'])
    ret_df.sort_values(by='pp_sq_m', ascending=True) #Sort by avg
 
    df_by_year = (df['sell_date'].year == inpyear)
    for entry in df_by_year:
        str_zip = entry['zip_code'][0:3]
        if str_zip in zipcodes:
            values_dict[str_zip][0] += entry['price_per_sq_m']
            values_dict[str_zip][1] += 1
    for nozip in zipcodes:
        myval = values_dict[nozip]
        ret_avg = myval[0] / myval[1]
        ret_df.append({'zip' : nozip , 'pp_sq_m' : ret_avg} , ignore_index=True)
    return ret_df

def calc_coords(row):
    row['sell_date'] = pd.to_datetime(row['sell_date'], format='%d%m%Y', errors='ignore')
    myVal = row['zip_code']
    narrowVals = myDict[myVal]
    for decoded_node in narrowVals:
        if decoded_node[0] == row['address']:
            row[lat] = decoded_node[1] 
            row[lon] = decoded_node[2]
            break;
        
def run():

    # Clean up for runing file repeatedly
    if os.path.isfile('./boliga.csv'):
        os.remove('boliga.csv') #Remove csv file so it doesn't append

    zf = zipfile.ZipFile('boliga_stats.zip', 'r')
    zf.extractall('boliga')

    path = r'C:\Users\seana\Desktop\Sem6\BI\BI\Assignment2\boliga'                     
    all_files = glob.glob(os.path.join(path, "*.csv"))    

    df_from_each_file = (pd.read_csv(f) for f in all_files)
    housing_df = pd.concat(df_from_each_file, ignore_index=True)
    housing_df.assign(lat="0",lon="0")  #Add empty new columns
    housing_df.apply(calc_coords, axis=1)
    zips_to_avg = [1049, 1050, 5000, 8000, 9000]
    avg_df_1992 = calc_price_averages(housing_df, zips_to_avg, 1992)
    avg_df_2016 = calc_price_averages(housing_df, zips_to_avg, 2016)
    
    # Gen csv files per city
    my_dfs = (avg_df_1992, avg_df_2016)
    csv_cph1 = zip_df_csv(my_dfs, 1049)
    csv_cph2 = zip_df_csv(my_dfs, 1050)
    csv_odense = zip_df_csv(my_dfs, 5000)
    csv_aarhus = zip_df_csv(my_dfs, 8000)
    csv_aalborg = zip_df_csv(my_dfs, 9000)

    # Make the dataframe into a csv
    export_csv = housing_df.to_csv (r'C:\Users\seana\Desktop\Sem6\BI\BI\Assignment2\boliga.csv', index = None, header=True)

    gen_scatter() #Generate scatter plot with coordinates

    # Create array of distances using formula
    dist_arr = distance_arr()
    shutil.rmtree('boliga') # Remove decompressed folder

if __name__ == '__main__':
    myDict = decode_node_to_csv()
    run()