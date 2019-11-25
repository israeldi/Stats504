import numpy as np
import statsmodels.api as sm
import pandas as pd
import json
import matplotlib.pyplot as plt

dir_ = "./Data/"
df = pd.read_csv(dir_+ "2017_utilization_reduced.csv.gz", 
                 dtype={"nppes_provider_zip": str,
                        "tract": str,
                        "county": str,
                        "cbsa": str})

df = df.loc[df.nppes_entity_code=="I", :]

# Include this as a covariate to proxy for practice size
df["total_npi_cnt"] = df.groupby("npi")["line_srvc_cnt"].transform(np.sum)
df["log_total_npi_cnt"] = np.log(df.total_npi_cnt)


fml = [
    "line_srvc_cnt ~ provider_type + log_total_npi_cnt + log_total_zip_cnt",
    "line_srvc_cnt ~ nppes_provider_state + log_total_npi_cnt + log_total_geo_cnt",
    "line_srvc_cnt ~ provider_type + nppes_provider_state + log_total_geo_cnt + log_total_npi_cnt"
]

'''
fml = [
    "line_srvc_cnt ~ provider_type + log_total_npi_cnt + log_total_zip_cnt",
    "line_srvc_cnt ~ nppes_provider_state + log_total_npi_cnt + log_total_geo_cnt",
    "line_srvc_cnt ~ C(provider_type) * log_total_geo_cnt + C(nppes_provider_state) * log_total_geo_cnt + log_total_npi_cnt"
]
'''

with open("hcpcs_description.json") as fr:
    hcpcs = json.load(fr)
    

#############################################################################
'''
1. Conduct an analysis of the Medicare claims data, focusing on geographical 
clustering of claims for specific procedures. 

2. We suggest that you use an appropriate GLM, and control for major 
covariates such as provider type and state, then use the residuals from these 
fits to understand geographic clustering that may be a result of unmeasured 
region-level covariates.

3. You should consider additional levels of geographical structure including: 
Census Tract, County, CBSA, and/or Other Levels.
'''

geo_struct = ["nppes_provider_zip", "tract", "county", "cbsa"]


mx = 2
result_dict = {}
results_df = pd.DataFrame()

# Filter Just two codes with a small number of instances for testing
# ds = df[df.loc[:,"hcpcs_code"] == "99203"]
num_inst = 25000
B = df.groupby("hcpcs_code").count()[["npi"]]
# filter_codes = B[(B["npi"] > 25000) & (B["npi"] < 50000)].index.tolist()[0:2]
filter_codes = B[(B["npi"] > num_inst)].index.tolist()
filter_ind = df.loc[:,"hcpcs_code"].isin(filter_codes)
filter_df = df[filter_ind].copy()
filter_df.to_csv(dir_ + "filter_df.csv", index=None)


for geo_group in geo_struct:
    scale = []
    icc = []
    codes = []
    
    ds = filter_df.copy()
    
    # Include this as a covariate to proxy for zip code attributes
    # related to population demographics
    ds["total_geo_cnt"] = ds.groupby(geo_group)["line_srvc_cnt"].transform(np.sum)
    ds["log_total_geo_cnt"] = np.log(ds.total_geo_cnt)
    ds = ds.dropna()
    
    # Testing with only few codes
    for code, dg in ds.groupby("hcpcs_code"):
    
        codes.append(code)
        
        model = sm.GEE.from_formula(fml[mx], family=sm.families.Poisson(), 
                                    groups= geo_group,
                                    cov_struct=sm.cov_struct.Exchangeable(), 
                                    data=dg)
        result = model.fit(maxiter=1, first_dep_update=0)
    
        # Estimate the scale parameter as if this were a quasi-Poisson model
        scale.append(np.mean(result.resid_pearson**2))
    
        icc.append(result.cov_struct.dep_params)
        print(code)
    
    hd = [hcpcs[c] for c in codes]
    
    rslt = pd.DataFrame({"code": codes, "icc": icc, "scale": scale, "description": hd})
    rslt = rslt.sort_values(by="icc")
    rslt["geo"] = geo_group
    # rslt.to_csv(dir_ + "model.csv", index=None)
    # result_dict[geo_group] = rslt
    # results_df = pd.DataFrame(columns=["code", "icc", "scale", "description"])
    results_df = results_df.append(rslt)

results_df.to_csv(dir_ + "model_additive.csv", index=None)
#results_df.pivot("code", "geo", "icc").plot(kind='bar')
#plt.savefig("hw4_barchart.png")


##############################################################################
# Plot Results
file = "model_additive.csv"
d_test = pd.read_csv(dir_ + file)

# Top 10 codes for zip
top_codes = d_test[d_test["geo"] == "nppes_provider_zip"].tail(8)["code"]
filter_ind = d_test.loc[:,"code"].isin(top_codes)
d_test = d_test[filter_ind]

# Plot bar chart
cols = [0, 1, 3, 2]
d_test.pivot("code", "geo", "icc").sort_values(by="nppes_provider_zip").iloc[:,cols].plot(kind='bar')
plt.legend(["CBSA", "County", "Tract", "Zip Code"])
plt.title("ICC Medicare Procedures by Geographical Structure")
plt.ylabel("ICC")
plt.tight_layout()
plt.savefig("hw4_barchart.png")

hd = [hcpcs[c] for c in top_codes]
descriptions = pd.DataFrame({"codes": top_codes, "Description": hd})
descriptions.to_csv(dir_ + "descriptions.csv", index=None)

# group_sizes = [len(df.groupby(geo)) for geo in geo_struct]
# out: [244243, 16247, 2969, 953]















