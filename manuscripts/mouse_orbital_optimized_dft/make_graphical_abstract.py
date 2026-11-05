"""Create the chemistry-centred graphical abstract for the IJQC submission."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

P = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=220)
ax.set_xlim(0, 10.5); ax.set_ylim(0, 5); ax.axis("off")
navy, teal, orange, purple, pale = "#173f56", "#168b8b", "#df7d35", "#7866a9", "#eef5f7"

def box(x, y, w, h, title, body, color=teal):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.08,rounding_size=0.10",
                                facecolor="white", edgecolor="#b7cbd2", lw=1.2))
    ax.plot([x+0.18,x+w-0.18],[y+h-0.13,y+h-0.13], color=color, lw=4, solid_capstyle="round")
    ax.text(x+w/2,y+h-0.35,title,ha="center",va="center",fontsize=12,fontweight="bold",color=color)
    ax.text(x+w/2,y+0.40,body,ha="center",va="center",fontsize=9.2,color=navy,linespacing=1.3)

ax.text(0.25,4.66,"STATE-SPECIFIC EXCITATIONS IN LARGE MOLECULES",fontsize=16,fontweight="bold",color=navy)
ax.text(0.27,4.36,"Orbital relaxation connects the absorption energy to the geometry, forces and vibrational curvature.",fontsize=9.7,color="#5b7180")

# molecular chromophore: conjugated chain with heteroatom donor
box(0.25,2.35,2.25,1.45,"Chromophore","π  →  π*   /   n  →  π*\ncharge redistribution",teal)
xs=np.linspace(.55,2.15,5); ys=np.array([1.95,2.18,1.95,2.18,1.95])
for i in range(4):
    ax.plot(xs[i:i+2],ys[i:i+2],color=navy,lw=2)
    if i%2==0: ax.plot(xs[i:i+2],ys[i:i+2]+.08,color=navy,lw=1)
ax.text(.48,1.69,"conjugated molecular skeleton",fontsize=8.3,color="#5b7180")

# orbital / stationary state panel
box(2.95,2.05,2.55,1.75,"Spin-pure state","common orbitals\nROKS singlet surface\nstationary saddle",orange)
arrow = lambda a,b,c=orange: ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.8,color=c))
arrow((2.58,3.05),(2.88,3.05))

# energy curves in lower middle
x=np.linspace(5.9,8.0,180); q=(x-6.95)
g=0.45+0.18*q*q
e=1.35+0.30*(q-0.25)**2
ax.plot(x,g,color=navy,lw=2.3); ax.plot(x,e,color=orange,lw=2.3)
ax.scatter([6.95],[g[np.argmin(abs(x-6.95))]],s=35,color=navy,zorder=3)
ax.scatter([7.2],[e[np.argmin(abs(x-7.2))]],s=35,color=orange,zorder=3)
ax.text(5.98,2.00,"state-specific energy surface",fontsize=9,fontweight="bold",color=purple)
ax.text(6.00,.34,"ground geometry",fontsize=7.5,color=navy)
ax.text(7.02,1.78,"relaxed excited geometry",fontsize=7.5,color=orange)
ax.text(6.00,.02,"nuclear coordinate",fontsize=8,color="#5b7180")

# spectrum panel
box(8.45,2.05,1.85,1.75,"Spectrum","",purple)
for xx,hh,col in [(8.78,.45,navy),(9.12,.78,orange),(9.46,1.22,orange),(9.80,.92,purple)]:
    ax.plot([xx,xx],[2.34,2.34+hh],lw=5,color=col,solid_capstyle="round")
ax.text(8.58,2.18,"state ordering",fontsize=7.3,color="#5b7180")
arrow((5.65,2.92),(8.38,2.92),purple)

# derivative outputs
for x0, title, body, col in [(2.9,"Energy","excitation gaps",teal),(5.0,"Gradient","relaxation\nforces",orange),(7.1,"Hessian","vibrations\ncurvature",purple)]:
    ax.add_patch(FancyBboxPatch((x0,.62),1.7,.78,boxstyle="round,pad=.06,rounding_size=.08",facecolor=pale,edgecolor="#b7cbd2",lw=1))
    ax.text(x0+.85,1.17,title,ha="center",fontsize=10.3,fontweight="bold",color=col)
    ax.text(x0+.85,.83,body,ha="center",fontsize=8.3,color=navy)
ax.plot([4.22,8.0],[1.52,1.52],color="#a9bdc5",lw=1.2,ls="--")
ax.text(.28,.18,"DFT-informed orbital optimization  |  spin-pure excited states  |  analytic nuclear response",fontsize=9.8,fontweight="bold",color=navy)
fig.savefig(P/"figures/graphical_abstract_ijqc.pdf",bbox_inches="tight")
fig.savefig(P/"figures/graphical_abstract_ijqc.png",bbox_inches="tight",dpi=220)
