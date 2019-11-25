"""
The data are available from the link below, use the "downloadable tab delimited"
link to get the data.

https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Physician-and-Other-Supplier2017.html

After downloading the zip file, unzip it.  Then recompress with gzip
the file called "Medicare_Provider_Util_Payment_PUF_CY2017.txt".  If
you decide not to gzip it, you will need to modify the code below.

To obtain additional levels of geographic stratification, you will
need to obtain crosswalk files from the links below.

https://www.huduser.gov/portal/datasets/usps_crosswalk.html

The zip-tract crosswalk is needed to run the script below.
"""

import pandas as pd
import numpy as np
# import gzip
import json
dir_ = "Data/"

# Geographical Structures
zip_tract = pd.read_excel(dir_ + "ZIP_TRACT_092019.xlsx", dtype={"zip": str})
zt = {a:b for a,b in zip(zip_tract.zip, zip_tract.tract)}

zip_county = pd.read_excel(dir_ + "ZIP_COUNTY_092019.xlsx", dtype={"zip": str})
zc = {a:b for a,b in zip(zip_county.zip, zip_county.county)}

zip_cbsa = pd.read_excel(dir_ + "ZIP_CBSA_092019.xlsx", dtype={"zip": str})
zb = {a:b for a,b in zip(zip_cbsa.zip, zip_cbsa.cbsa)}


##############################################################################
# The CMS utilization file name
util_file = dir_ + "Medicare_Provider_Util_Payment_PUF_CY2017.txt.gz"

# The file name to use for the output file.
out_file = dir_ + "2017_utilization_reduced.csv.gz"

usecols = ["npi", "nppes_entity_code", "nppes_provider_zip", "nppes_provider_state",
           "provider_type", "hcpcs_code", "hcpcs_description", "line_srvc_cnt"]
df = pd.read_csv(util_file, skiprows=[1], delimiter="\t", usecols=usecols,
                 dtype={"nppes_provider_zip": str})
dx = df.groupby(["npi", "hcpcs_code"]).agg({"nppes_entity_code": "first",
                                            "nppes_provider_zip": "first",
                                            "nppes_provider_state": "first",
                                            "provider_type": "first",
                                            "hcpcs_description": "first",
                                            "line_srvc_cnt": np.sum})
dx = dx.reset_index()
dx = dx.dropna()

dx["line_srvc_cnt"] = dx["line_srvc_cnt"].astype(np.int)

# Save the hcpcs descriptions as a map from hcpcs code to
# hcpcs description, since it takes up a lot of space to
# leave it as a column in the file.
x = dx.loc[:, ["hcpcs_code", "hcpcs_description"]]
x = x.set_index("hcpcs_code")
h = x.to_dict()["hcpcs_description"]
with open("hcpcs_description.json", "w") as fh:
    json.dump(h, fh)
dx = dx.drop("hcpcs_description", axis=1)

##############################################################################
# Add tract information
dx["tract"] = [str(zt.get(z[0:5], "")) for z in dx.nppes_provider_zip]
dx["county"] = [str(zc.get(z[0:5], "")) for z in dx.nppes_provider_zip]
dx["cbsa"] = [str(zb.get(z[0:5], "")) for z in dx.nppes_provider_zip]
dx = dx.dropna()

dx.to_csv(out_file, index=None, compression="gzip")

















