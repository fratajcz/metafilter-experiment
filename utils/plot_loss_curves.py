import json
import numpy as np




## TransE:
# Subset:
subset =  "/20210222-160447-hpo-CxD-drkg-subset-with-inverse-transe-both/00002/"
# Full:
full = "20210222-071446-hpo-CxD-drkg-full-with-inverse-transe-both/00002/"


runs = {'Hetionet':
            {'RESCAL': 
                {'subset': "20210212-062234-hpo-CxD-hetionet-fold1-subset-with-inverse-rescal-both/00019/",
                 'full': "20210212-062234-hpo-CxD-hetionet-fold1-full-with-inverse-rescal-both/00019/"},
             'DistMult':
                {'subset': "20210203-062208-hpo-CxD-hetionet-fold1-subset-with-inverse-distmult-both/00006/",
                 'full': "20210203-062637-hpo-CxD-hetionet-fold1-full-with-inverse-distmult-both/00006/"},
             'ComplEx':
                {'subset': "20210203-064415-hpo-CxD-hetionet-fold1-subset-with-inverse-complex-both/00006/",
                 'full': "20210203-064415-hpo-CxD-hetionet-fold1-full-with-inverse-complex-both/00006/"},
             'ConvE':
                 {'subset': "20210212-092340-hpo-CxD-hetionet-fold1-subset-with-inverse-conve-both/00025/",
                 'full': "20210212-071051-hpo-CxD-hetionet-fold1-full-with-inverse-conve-both/00018/"},   
                 },
        'DRKG':
            {'TransE': 
                {'subset': "/20210222-160447-hpo-CxD-drkg-subset-with-inverse-transe-both/00002/",
                 'full': "20210222-071446-hpo-CxD-drkg-full-with-inverse-transe-both/00002/"},
             'DistMult':
                {'subset': "20210302-113022-hpo-CxD-drkg-subset-with-inverse-distmult-both/00006/",
                 'full': "20210302-154646-hpo-CxD-drkg-full-with-inverse-distmult-both/00006/"},
             'ComplEx':
                {'subset': "20210302-170815-hpo-CxD-drkg-subset-with-inverse-complex-both/00006/",
                 'full': "20210302-171854-hpo-CxD-drkg-full-with-inverse-complex-both/00006/"},
             'ConvE':
                 {'subset': "",
                 'full': ""},   
             'RESCAL':
                {'subset': "",
                 'full': ""},
                }
            }

prefix = "~/kge/local/experiments/"
suffix = "trace.yaml"

for dataset, models in runs.items():
    for model, settings in models.items():
        
        subset = settings["subset"]
        full = settings["full"]
        
        if subset == "" or full == "":
            continue

        subset_loss = []
        subset_mrr = [0]
        full_loss = []
        full_mrr = [0]
        subset_timestamps = []
        full_timestamps = []
        subset_epoch_times = []
        full_epoch_times = []

        with open(prefix + subset + suffix, "r") as trace:
            for line in trace:
                entries = line[1:-2].split(",")
                entry = {entry.split(":")[0].strip(): entry.split(": ")[1].strip() for entry in entries}
                for key, value in entry.items():
                    if key.startswith("avg_loss"):
                        subset_loss.append(float(value))
                    if key.startswith("mrr_avg_val"):
                        subset_mrr.append(float(value))
                    if key.startswith("timestamp"):
                        subset_timestamps.append(float(value))
                    if key.startswith("epoch_time"):
                        subset_epoch_times.append(float(value))




        with open(prefix + full + suffix, "r") as trace:
            for line in trace:
                entries = line[1:-2].split(",")
                entry = {entry.split(":")[0].strip(): entry.split(": ")[1].strip() for entry in entries}
                for key, value in entry.items():
                    if key.startswith("avg_loss"):
                        full_loss.append(float(value))
                    if key.startswith("mrr_avg_val"):
                        full_mrr.append(float(value))
                    if key.startswith("timestamp"):
                        full_timestamps.append(float(value))
                    if key.startswith("epoch_time"):
                        full_epoch_times.append(float(value))

        subset_duration = subset_timestamps[-1] - subset_timestamps[0]
        full_duration = full_timestamps[-1] - full_timestamps[0]

        print("{} {} {}: {} hours total, {} m per epoch".format(dataset,model,"subset",round(subset_duration/3600,1),round(np.mean(subset_epoch_times)/60,1)))
        print("{} {} {}: {} hours total,  {} m per epoch".format(dataset,model,"full",round(full_duration/3600,1),round(np.mean(full_epoch_times)/60,1)))
        while len(full_loss) < len(subset_loss):
            full_loss.append(full_loss[-1])
            
        while len(full_mrr) < len(subset_mrr):
            full_mrr.append(full_mrr[-1])

        while len(subset_loss) < len(full_loss):
            subset_loss.append(subset_loss[-1])
            
        while len(subset_mrr) < len(full_mrr):
            subset_mrr.append(subset_mrr[-1])

        x =  [x + 1 for x in  range(len(full_loss))]

        evaluate_every_n = 2 if dataset == "Hetionet" else 3

        x_mrr = [x*evaluate_every_n for x in range(len(full_mrr))]

        best_mrr_subset_y = np.max(subset_mrr)
        best_mrr_subset_x = subset_mrr.index(best_mrr_subset_y) * evaluate_every_n

        best_mrr_full_y = np.max(full_mrr)
        best_mrr_full_x = full_mrr.index(best_mrr_full_y) * evaluate_every_n

        

        import matplotlib.pyplot as plt

        SMALL_SIZE = 6
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


        cm = 1/2.54  # centimeters in inches
        plt.xlim((x[0],x[-1]))
        fig, ax1 = plt.subplots(figsize=(8.5*cm, 7*cm), dpi=400)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.set_xlabel('Epochs')
        #ax1.set_title('Best Run {} {}'.format(dataset,model))
        ax1.set_title('')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('MRR')  # we already handled the x-label with ax1

        import matplotlib.lines as mlines

        blue_patch = mlines.Line2D([],[],color='blue', label='Training Loss Modified')
        lightblue_patch = mlines.Line2D([],[],color='lightblue', label='Validation MRR Modified')
        green_patch = mlines.Line2D([],[],color='green', label='Training Loss Original')
        lightgreen_patch = mlines.Line2D([],[],color='lightgreen', label='Validation MRR Original')

        plt.legend(handles=[blue_patch,lightblue_patch, green_patch, lightgreen_patch], loc="best")    
        
        plt.plot((best_mrr_subset_x, x[-1]), (best_mrr_subset_y, best_mrr_subset_y), linestyle='dashed', color="0.8")
        plt.plot((best_mrr_full_x, x[-1]), (best_mrr_full_y, best_mrr_full_y), linestyle='dashed', color="0.8")

        ax1.plot(x,subset_loss,c="blue")
        ax2.plot(x_mrr,subset_mrr,c="lightblue")
        ax1.plot(x,full_loss,c="green")
        ax2.plot(x_mrr,full_mrr,c="lightgreen")
        plt.plot(best_mrr_subset_x, best_mrr_subset_y, 'rx')
        plt.plot(best_mrr_full_x, best_mrr_full_y, 'rx')

        fig.tight_layout()

        plt.savefig("~/{}_{}_loss_vs_mrr.png".format(dataset, model), dpi=400)
