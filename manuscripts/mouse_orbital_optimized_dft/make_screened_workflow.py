"""Vector workflow of the implemented prepared semiempirical OO-CID surface."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(8,3.1));ax.set(xlim=(0,10),ylim=(-.18,4));ax.axis('off')
def box(x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.08',facecolor=color,edgecolor='#8097a5',lw=.9));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5,color='#143347')
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1,color='#476875'))
box(.15,2.65,2.75,1.05,'Native GFN2 engine\nEnergy, Fock, spin response\nand analytic nuclear primitives','#e9f1f7')
box(.15,.35,2.75,1.05,'One-time DFT preparation\nF/Q orbital-energy reference\n50/50; response lambda = 0.9','#e9f1f7')
box(3.5,1.45,2.8,1.15,'Common prepared energy\nStationary native ground\nSpin-pure ROKS saddle states','#dceff0')
box(7.0,2.65,2.75,1.05,'Frozen screened interaction\nFrontier + reflection-even\nsinglet double excitations','#e9f1f7')
box(7.0,1.2,2.75,.95,'Moving ground-orthogonal\ncoherent CSF eigenproblem','#dceff0')
box(4.9,-.01,4.85,.8,'Analytic gradients / Hessians + APOC / LINK\nNo runtime DFT or TDA solve; no energy offset','#d4e7eb')
arrow((2.98,3.12),(3.45,2.2));arrow((2.98,.85),(3.45,1.72));arrow((6.38,2.02),(6.93,1.7));arrow((8.37,2.57),(8.37,2.23));arrow((8.37,1.13),(8.37,.86))
fig.tight_layout(pad=.4);fig.savefig(P/'figures/screened_method_workflow.pdf',bbox_inches='tight')
