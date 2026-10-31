"""Publication figures from archived data and labelled analytical schematics.
No electronic calculations, optimization, or Hessian construction is performed.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.colors import to_rgb,LinearSegmentedColormap
from matplotlib.ticker import LogLocator,NullFormatter
P=Path(__file__).resolve().parent;OUT=P/'figures';OUT.mkdir(exist_ok=True)
D=json.loads((P/'data/SUMMARY.json').read_text())
NAVY='#173D52';TEAL='#168A8A';ORANGE='#D98745';PURPLE='#8877AE';GRAY='#75838E';LIGHT='#E7EEF1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.edgecolor':'#A3B0B8','axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GRAY,'ytick.color':GRAY,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
def save(fig,name):
 for ext in ['pdf','svg']:fig.savefig(OUT/(name+'.'+ext),facecolor='white',bbox_inches='tight',pad_inches=.12)
 fig.savefig(OUT/(name+'.png'),dpi=240,facecolor='white',bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
def title(ax,letter,text):ax.set_title(letter+'  '+text,loc='left',pad=12,color=NAVY)
def box(ax,x,y,w,h,head,body,color=TEAL,fs=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.13',facecolor='white',edgecolor='#CCD8DE',lw=1))
 ax.plot([x+.16,x+w-.16],[y+h-.13]*2,color=color,lw=3,solid_capstyle='round')
 ax.text(x+w/2,y+h-.42,head,ha='center',va='center',fontsize=fs,fontweight='bold',color=color)
 ax.text(x+w/2,y+.38*h,body,ha='center',va='center',fontsize=fs-1.1,linespacing=1.5,color=NAVY)
def arrow(ax,a,b,color=GRAY,style='-|>',rad=0):ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=13,color=color,lw=1.4,connectionstyle=f'arc3,rad={rad}'))
# 1. Actual implemented workflow, expressed as vector graphics.
fig,ax=plt.subplots(figsize=(9.5,5.1));ax.set(xlim=(0,12),ylim=(-.5,6.2));ax.axis('off')
ax.text(.15,5.98,'MOUSE  /  SEMIEMPIRICAL ORBITAL-OPTIMIZED DFT',fontsize=12,fontweight='bold')
box(ax,.2,3.8,3.0,1.6,'DFT reference','PBE0 at the anchor geometry\nvalence charge + spin projection',TEAL,fs=9.1)
box(ax,.2,.9,3.0,1.9,'Native GFN2 engine','Energy, Fock and spin response\nmetric / Pulay terms\nfirst + second nuclear derivatives',NAVY,fs=8.6)
box(ax,4.1,4.1,3.05,1.3,'Parent F','Full projected Fock correction',TEAL)
box(ax,4.1,2.35,3.05,1.3,'Parent Q','Qualified DFT orbital spacings',PURPLE)
arrow(ax,(3.25,4.72),(4.03,4.72));arrow(ax,(3.22,4.0),(4.02,3.01))
ax.text(5.62,3.87,'same charge / spin kernel',fontsize=8.3,ha='center',color=GRAY)
box(ax,8.15,2.65,3.55,2.35,'Stationary singlet energy','Two spin-pure ROKS CSFs\nfixed 50/50 mean\nrelative active-pair constraints',ORANGE,fs=10.4)
arrow(ax,(7.2,4.72),(8.1,4.23));arrow(ax,(7.2,3.0),(8.1,3.35))
ax.text(8.85,2.10,'KKT response',ha='center',fontsize=9,fontweight='bold',color=ORANGE)
arrow(ax,(9.92,2.6),(9.92,1.8),ORANGE)
for x,h,b,c in [(4.1,'Energy',r'$E_{\mathrm{exc}}$',TEAL),(6.72,'Gradient',r'$dE / dR$',TEAL),(9.34,'Hessian',r'$d^2E / dR^2$',ORANGE)]:
 box(ax,x,.25,2.3,1.3,h,b,c,fs=10)
ax.plot([3.25,10.49],[1.8,1.8],color=GRAY,lw=1.2)
for xx in [5.25,7.87,10.49]:arrow(ax,(xx,1.8),(xx,1.59),ORANGE)
ax.text(.2,-.18,'Frozen reference operators  |  geometry-dependent orbital relaxation  |  no additive excitation-energy offset',fontsize=8.4,color=GRAY)
save(fig,'method_workflow')
# 2. Explicitly schematic electronic saddle versus a nuclear minimum.
fig=plt.figure(figsize=(8.5,4.7));gs=fig.add_gridspec(2,2,height_ratios=[1,.31],left=.06,right=.97,bottom=.08,top=.90,wspace=.30,hspace=.12)
a=fig.add_subplot(gs[0,0],projection='3d',computed_zorder=False);v=np.linspace(-1,1,46);x,y=np.meshgrid(v,v);z=.5*(x*x-y*y)
cmap=LinearSegmentedColormap.from_list('saddle',[PURPLE,'#F6F8F9',TEAL]);a.plot_surface(x,y,z,cmap=cmap,alpha=.93,linewidth=0,antialiased=True)
a.plot([-1,1],[0,0],[.5,.5],alpha=0) # fix bounds without data interpretation
u=np.linspace(-1,1,130);a.plot(u,0*u,.5*u*u,color=TEAL,lw=2.4,zorder=8);a.plot(0*u,u,-.5*u*u,color=PURPLE,lw=2.4,zorder=9);a.scatter([0],[0],[0],s=27,color=NAVY,depthshade=False,zorder=10)
a.view_init(elev=25,azim=-57);a.set(xlabel=r'Orbital coordinate $\kappa_1$',ylabel=r'Orbital coordinate $\kappa_2$',zlabel='Electronic energy')
a.set_xticks([]);a.set_yticks([]);a.set_zticks([]);a.grid(False)
for axis in [a.xaxis,a.yaxis,a.zaxis]:axis.pane.fill=False;axis.line.set_color('#CDD6DB')
a.set_title('a  Electronic stationarity',loc='left',pad=12,color=NAVY,fontweight='bold')
b=fig.add_subplot(gs[0,1]);rr=np.linspace(-1.3,1.3,200);ee=.5*(1-.45**2)*rr**2;b.plot(rr,ee,color=ORANGE,lw=3);b.scatter([0],[0],s=45,color=NAVY,zorder=4);b.axhline(0,color='#C5D0D6',lw=.8,zorder=0)
b.annotate('Nuclear minimum',xy=(0,0),xytext=(.17,.28),arrowprops=dict(arrowstyle='-',color=GRAY),fontsize=9,color=NAVY)
b.set(xlabel=r'Nuclear displacement $R$',ylabel='Relaxed state energy',ylim=(-.06,.85));b.set_xticks([]);b.set_yticks([]);title(b,'b','Relaxed nuclear surface')
c=fig.add_subplot(gs[1,:]);c.axis('off');c.add_patch(FancyBboxPatch((.02,.15),.96,.75,transform=c.transAxes,boxstyle='round,pad=.015',facecolor='#F0F5F7',edgecolor='none'))
c.text(.5,.65,r'$\mathcal{L}_{\kappa}=0,\quad g=0\qquad\Longrightarrow\qquad H=\mathcal{L}_{RR}-B^{\mathsf{T}}\mathcal{K}^{-1}B$',ha='center',va='center',transform=c.transAxes,fontsize=13,color=NAVY)
c.text(.5,.27,'Schematic local quadratic model; the electronic saddle index depends on the targeted state.',ha='center',va='center',transform=c.transAxes,fontsize=8,color=GRAY)
save(fig,'stationary_response')
# 3. Recorded vertical energies, including prospectively labelled comparisons.
rows=D['regression_states']+D['prospective_states'];pros=D['prospective_states']
fig,ax=plt.subplots(1,2,figsize=(8.8,3.9),gridspec_kw={'width_ratios':[1,1.25]});fig.subplots_adjust(left=.075,right=.975,bottom=.17,top=.83,wspace=.42)
a,b=ax
for lo,hi,color,label,marker in [(0,15,TEAL,'Exposed regression','o'),(15,19,ORANGE,'Prospective','D')]:
 q=rows[lo:hi];a.scatter([r['tbe_ev'] for r in q],[r['excitation_ev'] for r in q],s=32,c=color,marker=marker,edgecolor='white',lw=.6,label=label,zorder=4)
a.plot([1,7],[1,7],ls=(0,(3,3)),color='#A6B1B8',lw=1,zorder=1);a.set(xlabel='Reference TBE (eV)',ylabel='MOUSE OO-DFT (eV)',xlim=(1,7),ylim=(1,7));a.set_aspect('equal',adjustable='box');a.legend(frameon=False,loc='upper left',fontsize=7.3,handletextpad=.3);title(a,'a','Vertical excitation energies')
a.text(.98,.04,'19 states\nMAE 0.183 eV',ha='right',va='bottom',transform=a.transAxes,fontsize=9,fontweight='bold',color=TEAL)
order=['cyanoformaldehyde','cyclopropenone','pyridazine','imidazole'];pp={r['molecule']:r for r in pros};yy=np.arange(4)
ev=[pp[n]['excitation_ev']-pp[n]['tbe_ev'] for n in order]
b.hlines(yy,0,ev,color=TEAL,alpha=.35,lw=1.6);b.scatter(ev,yy,c=TEAL,marker='s',s=38,zorder=4,label='OO-DFT',edgecolor='white',lw=.45)
b.axvline(0,color='#9AA9B2',lw=.8);b.set(xlabel='Signed energy error (eV)',xlim=(-.65,.4),ylim=(3.55,-.62));b.set_yticks(yy,['Cyanoformaldehyde','Cyclopropenone','Pyridazine','Imidazole'],fontsize=8);b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False);b.grid(axis='x',color='#EDF1F3',lw=.7,zorder=0);title(b,'b','Prospective OO-DFT errors')
fig.text(.65,.035,'OO-DFT MAE: 0.304 eV',fontsize=9,color=TEAL,fontweight='bold')
save(fig,'vertical_validation')
# 4. Ball-and-stick views of actual optimized coordinates; distance-defined bonds.
colors={'C':'#4E626D','N':'#497DB8','O':'#D85C55','H':'#ECF1F3'};radii={'C':.76,'N':.71,'O':.66,'H':.31};sizes={'C':.24,'N':.24,'O':.24,'H':.145}
def lighten(c,f):return tuple((1-f)*np.array(to_rgb(c))+f)
def sphere(ax,x,y,r,color,zorder):
 ax.add_patch(Circle((x+.025,y-.035),r,facecolor='#000000',alpha=.07,edgecolor='none',zorder=zorder-.1))
 for k in range(18):
  f=k/17;rr=r*(1-.70*f);dx=-.10*r*f;dy=.18*r*f
  ax.add_patch(Circle((x+dx,y+dy),rr,facecolor=lighten(color,.02+.42*f),edgecolor='none',zorder=zorder+k*.001))
 ax.add_patch(Circle((x,y),r,facecolor='none',edgecolor=lighten(color,-.12),lw=.45,zorder=zorder+.1))
def molecule(ax,name):
 lines=(P/'coordinates'/f'{name}_minimum.xyz').read_text().splitlines()[2:];atoms=[s.split()[0] for s in lines];xyz=np.array([[float(v) for v in s.split()[1:]] for s in lines]);xyz-=xyz.mean(axis=0)
 # Principal-axis orientation is a rigid display rotation, not a new geometry.
 _,_,vh=np.linalg.svd(xyz,full_matrices=False);xyz=xyz@vh.T
 xyz[:,0]*=-1 if xyz[np.argmax(np.abs(xyz[:,0])),0]<0 else 1
 # Small tilt reveals pyramidalization while retaining readable bonds.
 angle=np.deg2rad(20);rot=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]]);xyz=xyz@rot.T
 bonds=[]
 for i in range(len(atoms)):
  for j in range(i):
   d=np.linalg.norm(xyz[i]-xyz[j])
   if .4<d<radii[atoms[i]]+radii[atoms[j]]+.25:bonds.append((i,j))
 for i,j in sorted(bonds,key=lambda ij:float((xyz[ij[0],2]+xyz[ij[1],2])/2)):
  a,b=xyz[i],xyz[j];mid=.5*(a+b)
  ax.plot([a[0],b[0]],[a[1],b[1]],color='#BBC7CD',lw=5.5,solid_capstyle='round',zorder=2)
  ax.plot([a[0],mid[0]],[a[1],mid[1]],color=lighten(colors[atoms[i]],.12),lw=3.4,solid_capstyle='round',zorder=2.1)
  ax.plot([b[0],mid[0]],[b[1],mid[1]],color=lighten(colors[atoms[j]],.12),lw=3.4,solid_capstyle='round',zorder=2.1)
 for i in np.argsort(xyz[:,2]):sphere(ax,xyz[i,0],xyz[i,1],sizes[atoms[i]],colors[atoms[i]],3+int(np.where(np.argsort(xyz[:,2])==i)[0][0])*.2)
 extent=max(np.ptp(xyz[:,0])/2+.5,np.ptp(xyz[:,1])/2+.5,1.15);center=(xyz[:,:2].min(axis=0)+xyz[:,:2].max(axis=0))/2
 ax.set(xlim=(center[0]-extent,center[0]+extent),ylim=(center[1]-extent,center[1]+extent));ax.set_aspect('equal');ax.axis('off')
 vals=[]
 for i,j in bonds:
  if set([atoms[i],atoms[j]])==set(['C','O']):vals.append(np.linalg.norm(xyz[i]-xyz[j]))
 label='C-O  '+ ' / '.join(f'{v:.3f}' for v in sorted(vals))+' Å' if vals else 'N-O  '+f"{np.linalg.norm(xyz[atoms.index('N')]-xyz[atoms.index('O')]):.3f}"+' Å'
 return label
fig=plt.figure(figsize=(8.4,5.7));gs=fig.add_gridspec(2,6,left=.02,right=.98,bottom=.20,top=.88,hspace=.75,wspace=.25)
order=['nitroxyl','acetaldehyde','acetone','formamide','maleimide'];md={r['molecule']:r for r in D['molecular_minima']}
for k,n in enumerate(order):
 row=0 if k<3 else 1;lo=k*2 if k<3 else 1+(k-3)*2;ax=fig.add_subplot(gs[row,lo:lo+2]);bl=molecule(ax,n)
 ax.text(.5,1.13,chr(97+k)+'  '+n.capitalize(),ha='center',va='bottom',transform=ax.transAxes,fontsize=10.4,fontweight='bold')
 ax.text(.5,-.075,bl,ha='center',va='top',transform=ax.transAxes,fontsize=8.6,color=GRAY)
 ax.text(.5,-.22,r'$\nu_{\min}$'+f"  {min(md[n]['frequencies_cm']):.0f}"+r' cm$^{-1}$',ha='center',va='top',transform=ax.transAxes,fontsize=9,color=TEAL,fontweight='bold')
fig.text(.5,.025,'Optimized excited-state minima  •  positive projected vibrational curvatures',ha='center',fontsize=9.5,color=NAVY)
for k,s in enumerate(['C','N','O','H']):
 fig.text(.34+k*.09,.965,'●',color=colors[s],fontsize=11);fig.text(.362+k*.09,.968,s,fontsize=8.7,color=GRAY,va='center')
save(fig,'molecular_minima')
# 5. Measured derivative validation. Numerical checks are not production Hessians.
fig,ax=plt.subplots(1,2,figsize=(8.4,3.65),gridspec_kw={'width_ratios':[1,1]});fig.subplots_adjust(left=.10,right=.96,bottom=.26,top=.82,wspace=.60)
cols=[TEAL,ORANGE,PURPLE,NAVY,'#9A864D']
for col,n in zip(cols,order):
 r=md[n];ch=sorted(r['checks'],key=lambda q:q['step']);ax[0].loglog([q['step'] for q in ch],[q['hvp_error'] for q in ch],'-o',color=col,lw=1.5,ms=4,label=n.capitalize())
ax[0].set_xticks([1e-4,2e-4],[r'$1\times10^{-4}$',r'$2\times10^{-4}$']);ax[0].xaxis.set_minor_formatter(NullFormatter());ax[0].set_xlim(8.8e-5,2.25e-4);ax[0].set(xlabel='Finite-difference displacement (bohr)',ylabel=r'Max. HVP error ($E_{\rm h}$/bohr$^2$)');ax[0].grid(axis='y',color=LIGHT,lw=.7);title(ax[0],'a','Directional Hessian checks');fig.legend(*ax[0].get_legend_handles_labels(),frameon=False,fontsize=7.7,loc='lower center',bbox_to_anchor=(.5,.005),ncol=5)
checks={r['molecule']:r for r in D['full_hessian_column_checks']};ys=[0,1]
for y,n,col in zip(ys,['nitroxyl','formamide'],[TEAL,PURPLE]):
 r=checks[n];ax[1].hlines(y,1e-10,r['hessian_max_error'],color=col,lw=2,alpha=.35);ax[1].scatter(r['hessian_max_error'],y,s=55,color=col,zorder=4,edgecolor='white',lw=.8)
 ax[1].annotate(r"$"+f"{r['hessian_max_error']:.2e}".split("e")[0]+r"\times10^{"+str(int(f"{r['hessian_max_error']:.2e}".split("e")[1]))+r"}$",xy=(r['hessian_max_error'],y),xytext=(0,13),textcoords='offset points',ha='center',fontsize=8.2,fontweight='bold',color=col)
ax[1].set_xscale('log');ax[1].set(xlim=(1e-10,4e-8),ylim=(1.6,-.7),xlabel=r'Max. matrix error ($E_{\rm h}$/bohr$^2$)');ax[1].set_yticks(ys,['Nitroxyl\n9 columns','Formamide\n18 columns'],fontsize=8.4);ax[1].tick_params(axis='y',length=0);ax[1].spines['left'].set_visible(False);ax[1].grid(axis='x',color=LIGHT,lw=.7);title(ax[1],'b','Complete Cartesian Hessians');ax[1].text(.5,.05,r'$h=10^{-4}$ bohr',ha='center',transform=ax[1].transAxes,color=GRAY,fontsize=8.5)
save(fig,'hessian_validation')
manifest={'source_script':Path(__file__).name,'data_sha256':hashlib.sha256((P/'data/SUMMARY.json').read_bytes()).hexdigest(),'figures':{'method_workflow':'Vector diagram of the implemented method; no numerical results.','stationary_response':'Labelled schematic: E=0.5*x^2-0.5*y^2+0.5*R^2+0.45*R*x; local elimination gives Erelaxed=0.5*(1-0.45^2)*R^2. Not molecular data.','vertical_validation':'All 19 recorded energies; four prospective errors and MAEs; exact molecule identities from JSON.','molecular_minima':'Archived optimized XYZ geometries; rigid principal-axis display rotation and tilt; distance-inferred bonds, no bond-order assignment. Frequencies from SUMMARY.json.','hessian_validation':'Archived two-step HVP checks and all-column Hessian discrepancies; no new derivative evaluations.'}}
(OUT/'FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2))
print('Created five figures as PDF, SVG and PNG from archived results and labelled schematics.')
