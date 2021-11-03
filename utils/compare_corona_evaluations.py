import pandas as pd
import numpy as np

full = pd.read_csv("corona_mrrs_full.csv",index_col=0)
subset = pd.read_csv("corona_mrrs_subset.csv",index_col=0)


print(1/full["Disease::SARS-CoV2Spike"])
print(1/subset["Disease::SARS-CoV2Spike"])

print(np.mean(1/full["Disease::SARS-CoV2Spike"]))
print(np.mean(1/subset["Disease::SARS-CoV2Spike"]))


print(1/full.loc["Compound::DB12466", "Disease::SARS-CoV2nsp12"])
print(1/subset.loc["Compound::DB12466", "Disease::SARS-CoV2nsp12"])

print("Favipravir:")
favipravir = (1/subset.loc["Compound::DB12466",:])-(1/full.loc["Compound::DB12466", :])
favipravir_df = pd.concat([1/full.loc["Compound::DB12466",:], 1/subset.loc["Compound::DB12466", :], favipravir], axis=1)
print(favipravir)

print("Danoprevir:")
danoprevir = (1/subset.loc["Compound::DB11779",:])-(1/full.loc["Compound::DB11779", :])
danoprevir_df = pd.concat([1/full.loc["Compound::DB11779",:], 1/subset.loc["Compound::DB11779", :], danoprevir], axis=1)

print(danoprevir)

print("Lopinavir (ineffective):")
lopinavir = (1/subset.loc["Compound::DB01601",:])-(1/full.loc["Compound::DB01601", :])
lopinavir_df = pd.concat([1/full.loc["Compound::DB01601",:], 1/subset.loc["Compound::DB01601", :], lopinavir], axis=1)
print(lopinavir)

print("Ritonavir (ineffective):")
ritonavir = (1/subset.loc["Compound::DB00503",:])-(1/full.loc["Compound::DB00503", :])
ritonavir_df = pd.concat([1/full.loc["Compound::DB00503",:], 1/subset.loc["Compound::DB00503", :], ritonavir], axis=1)
print(ritonavir)

dictionary = {x: np.mean(y) for x,y in zip(["Favipravir","Danoprevir","Lopinavir","Ritonavir"],[favipravir,danoprevir,lopinavir,ritonavir])}

print(dictionary)

for name, df in zip(["Favipravir","Danoprevir","Lopinavir","Ritonavir"],[favipravir_df,danoprevir_df,lopinavir_df,ritonavir_df]):
    df.columns = ["Full","Subset","Change"]
    df.loc['Mean'] = df.mean()
    df.astype(int).to_csv(name + ".tsv", sep="\t")

multi_df = pd.concat([favipravir_df,danoprevir_df,lopinavir_df,ritonavir_df],axis=1)
multi_df.sort_index(inplace=True)
multi_df.astype(int).to_csv("all_four_antiviral_compounds.tab",sep= "&")

