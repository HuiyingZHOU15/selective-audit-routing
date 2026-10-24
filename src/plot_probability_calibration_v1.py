"""Plot SEC-BERT calibration reliability and proper-score changes."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures" / "sec_bert_probability_calibration_journal_v1"

def main():
    bins = pd.read_csv(TABLES / "sec_bert_reliability_bins_v1.csv")
    metrics = pd.read_csv(TABLES / "sec_bert_calibration_metrics_v1.csv")
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8.2,"axes.labelsize":9,"axes.titlesize":9,"xtick.labelsize":7.5,
        "ytick.labelsize":7.5,"legend.fontsize":7.2,"axes.linewidth":0.7,
        "pdf.fonttype":42,"ps.fonttype":42})
    navy, teal, sky, slate = "#173B63", "#00796B", "#56B4E9", "#64748B"
    fig, axes = plt.subplots(1, 2, figsize=(7.15,3.35), constrained_layout=True)

    ax = axes[0]
    styles = {"raw":(navy,"o","-","Raw probabilities"),
              "temperature_scaled":(teal,"s","--","Temperature scaled")}
    for method,(color,marker,ls,label) in styles.items():
        g = bins[(bins.method==method)&(bins.samples>0)].sort_values("mean_confidence")
        sizes = 22 + 85*np.sqrt(g.samples/g.samples.max())
        ax.plot(g.mean_confidence,g.accuracy,color=color,ls=ls,lw=1.1,alpha=.88,zorder=2)
        ax.scatter(g.mean_confidence,g.accuracy,s=sizes,marker=marker,color=color,
                   edgecolor="white",lw=.7,label=label,zorder=3)
    ax.plot([0,1],[0,1],color=slate,ls=(0,(3,2)),lw=.9,label="Perfect calibration")
    ax.set(xlim=(.30,1.02),ylim=(.20,1.24),xlabel="Mean predicted confidence",
           ylabel="Empirical accuracy")
    ax.set_yticks(np.arange(.2,1.01,.2))
    ax.set_title("Reliability on the calibration split",loc="left",fontweight="bold")
    ax.legend(loc="upper right",ncol=2,frameon=True,
              fancybox=False,facecolor="white",edgecolor="#D7DEE6",framealpha=.97,
              columnspacing=.9,handlelength=2)
    ax.text(.315,.225,"Marker area scales with bin sample size",fontsize=6.8,color=slate,
            bbox=dict(facecolor="white",edgecolor="none",alpha=.88,pad=1.1))

    ax = axes[1]
    cols, labels = ["nll","multiclass_brier","ece"],["NLL","Multiclass\nBrier","ECE (10 bins)"]
    raw = metrics[metrics.method=="raw"].iloc[0]
    scaled = metrics[metrics.method=="temperature_scaled"].iloc[0]
    rv, sv = np.array([raw[c] for c in cols],float), np.array([scaled[c] for c in cols],float)
    x,w = np.arange(3),.34
    br = ax.bar(x-w/2,rv,w,color=sky,edgecolor=navy,lw=.7,label="Raw")
    bs = ax.bar(x+w/2,sv,w,color="white",edgecolor=teal,lw=1,hatch="///",label="Temperature scaled")
    for bars in (br,bs):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.006,f"{bar.get_height():.3f}",
                    ha="center",va="bottom",fontsize=7,color=navy)
    for i,imp in enumerate(100*(rv-sv)/rv):
        ax.text(i,max(rv[i],sv[i])+.035,f"−{imp:.1f}%",ha="center",va="bottom",
                fontsize=7.2,color=teal,fontweight="bold")
    ax.set(xticks=x,xticklabels=labels,ylim=(0,.39),ylabel="Metric value (lower is better)")
    ax.set_title("Calibration metrics",loc="left",fontweight="bold")
    ax.legend(loc="upper right",ncol=2,frameon=True,fancybox=False,facecolor="white",
              edgecolor="#D7DEE6",framealpha=.97,columnspacing=.9,handlelength=1.7)

    for panel,ax in zip(("a","b"),axes):
        ax.text(-.15,1.08,panel,transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",color="#D7DEE6",lw=.45,alpha=.75); ax.set_axisbelow(True)
        ax.tick_params(width=.65,length=3)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"),dpi=450,bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__": main()
