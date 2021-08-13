import matplotlib.pyplot as plt
from numpy import save, load, nan, arange, mean, nanmax,nanmin
import matplotlib.animation as animation
from esn_dev.utils import score_over_time
import cmocean.cm as cmo


def animate_comparison(targets,predictions,kuro=True,filepath='comparison.mp4',fps=24,dpi=300, v=(None,None)):
    targets    = targets.copy()
    predictions= predictions.copy()
    if v[0] is None:
        #min/max
        vmin = min(targets.min(),predictions.min())
        vmax = max(targets.max(),predictions.max())
    else:
        vmin = v[0]
        vmax = v[1]

    if kuro:
        mask=load('Kuroshio_mask.npy')
        if targets[0].shape!=mask.shape:
            print('Not Kuroshimo-shaped')

        else:
            targets[:,mask]=nan
            predictions[:,mask]=nan

    def init():
        fig.suptitle(f't=0',y=1.2,fontsize='x-large')
        im1.set_data(targets[0],)
        im2.set_data(predictions[0])
        return (im1,im2)

    # animation function. This is called sequentially
    def animate(i):
        fig.suptitle(f't={i:03d}',y=1.2,fontsize='x-large')
        im1.set_data(targets[i])
        im2.set_data(predictions[i])
        return (im1,im2)

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(8.5,4),dpi=dpi,sharey=True)
    fig.tight_layout()
    im1 = ax1.imshow(targets[0,:,:],cmap=cmo.deep_r,origin='lower',vmin=vmin,vmax=vmax)
    im2 = ax2.imshow(predictions[0,:,:],cmap=cmo.deep_r,origin='lower',vmin=vmin,vmax=vmax)
    #[left, bottom, width, height] 
    cbar_ax = fig.add_axes([0.92, 0.235, 0.03, 0.58])
    fig.colorbar(im1, cax=cbar_ax)
    ax1.set_title('Targets')
    ax2.set_title('Predictions')
    ax1.grid(True,color='white')
    ax2.grid(True,color='white')
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    ax1.set_xlabel('Indices')
    ax2.set_xlabel('Indices')
    ax1.set_ylabel('Indices')
    
    #clear whitespace
    fig.subplots_adjust(
        left=0.09, 
        bottom=0.05, 
        right=0.9, 
        top=1, 
        wspace=0.05, 
        hspace=None)   
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=targets.shape[0], interval=20, blit=True)
    
    anim.save(
        filepath,
        writer=animation.FFMpegWriter(fps=fps),
        dpi=dpi,
        )
    
    return

def animate_this(anim_data, filepath='animation.mp4',fps=24,dpi=150):
    def init():
        im.set_data(anim_data[0])
        return (im,)

    # animation function. This is called sequentially
    def animate(i):
        data_slice = anim_data[i]
        im.set_data(data_slice)
        return (im,)
    fig, ax = plt.subplots()
    vmin = anim_data.min()
    vmax = anim_data.max()
    im = ax.imshow(anim_data[0],cmap='inferno',vmin=vmin,vmax=vmax,origin='lower')
    #plt.colorbar(im)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=anim_data.shape[0], interval=20, blit=True)
    #clear whitespace
    
    fig.subplots_adjust(
        left=0, 
        bottom=0, 
        right=1, 
        top=1, 
        wspace=None, 
        hspace=None)
    
    anim.save(
        filepath,
        writer=animation.FFMpegWriter(fps=fps),
        dpi=dpi,
        )
    
    return anim


def MSE_over_time(targets,predictions,subplot_kw=None):
    MSE_over_time = score_over_time(predictions,targets)
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw=subplot_kw,
        figsize=(5.5,2.74)
    )
    ax.plot(
        arange(1,len(targets)+1),
        MSE_over_time,
        color='black',
        linestyle='-',
        marker='.',
        markersize = 2,
        mec='red',
        mfc='red',
    )
    #standard labels if not set
    if len(ax.get_xlabel())<1:
        ax.set_xlabel('Time Step')
    if len(ax.get_ylabel())<1:
        ax.set_ylabel('MSE')
    
    return fig

def multiplot(data,framelist,
              plotname=None, 
              colorbar_kw = None,
              subplot_kw = None,
              fig_kw = None,
              gridspec_kw = None,axis_on=False,targets=None,timeunit=None):
    """
    Use case:
    Show a number of frames from a video sequence
    
    Create a plot with default of 4 rows
    and 2 cols from a 3d volume 'data'
    with slices in 'framelist'
    
    Example keywords that can be passed:
    (see plt.subplots() documentation)
    
    # fewer rows
    fig_kw=dict(nrows=3,ncols=2)) 
    # No spacing
    gridspec_kw = dict(hspace=0,wspace=0)
    # other colors
    colorbar_kw = dict(vmin=-10, cmap='viridis')
    # Share x axes
    subplot_kw = dict(xlim = (0,10))
    """
    # want plots sorted in time
    framelist.sort()
    if timeunit is None:
        namelist = framelist
    else:
        namelist = []
        # Expect timeunit format "3 days" or "5days"
        time_res = str(timeunit[0])
        for i in range(len(framelist)):
            namelist.append(r"$t=t_0$ +"+f" {(int(framelist[i]+1)*int(time_res))} {timeunit[1:]}")

    
    def override_kw(default_kw,custom_kw):
        # override default settings with
        # any custom ones
        try:
            for key in default_kw:
                if key not in custom_kw:
                    # use default if not specified by user
                    custom_kw[key]=default_kw[key]
        except:
            print('Using default k.w. args for one dict')
            custom_kw=default_kw
        return custom_kw
    
    # figure settings
    fig_default = dict(dpi=600,nrows = 4, ncols=2)
    fig_kw = override_kw(fig_default,fig_kw)
    
    # colorbar kws
    # including vmin, vmax, cmap
    #min/max of colorbar
    vmin = nanmin(data)
    vmax = nanmax(data)
    colorbar_default = dict(
        orientation='horizontal',
        cmap =cmo.deep_r,
        vmin = vmin,
        vmax = vmax,
    )
    colorbar_kw = override_kw(colorbar_default,colorbar_kw)
    
    # Grid specifications
    gridspec_default = dict(hspace=0.01*fig_kw['nrows'],wspace=0.02)
    gridspec_kw = override_kw(gridspec_default, gridspec_kw)

    #subplots settings
    subplots_default = dict()
    subplot_kw=override_kw(subplots_default,subplot_kw)
    
    # determine figure size
    hw_ratio = data[0].shape[1]/data[0].shape[0]
    figy =  2.625*fig_kw['nrows']
    figx  = 2.625*fig_kw['ncols']*hw_ratio

    fig, axes = plt.subplots(
        figsize=(figx,figy),
        gridspec_kw=gridspec_kw,
        subplot_kw=subplot_kw,
        **fig_kw,
    )
    axs = axes[:]
    axes = axes.reshape(-1)
    if targets is None:
        assert len(axes) == len(framelist)
        for i, ax in enumerate(axes):
            im = ax.imshow(data[framelist[i]],origin='lower',
                           cmap = colorbar_kw['cmap'],
                           vmin = colorbar_kw['vmin'],
                           vmax = colorbar_kw['vmax'])
            time_num = framelist[i]
            time_str = ''+f'{namelist[i]}'+''
            ax.set_title(time_str,fontsize='x-large')
            if axis_on==False:
                ax.axis('off') # disable index etc
        del colorbar_kw['cmap'] #not for use in colorbar()
        del colorbar_kw['vmin']
        del colorbar_kw['vmax']

        fig.subplots_adjust(bottom=0.02)
        #[left, bottom, width, height]
        cbar_ax = fig.add_axes([0.125, 0., .775, 0.025])
        cbar = fig.colorbar(im, cax=cbar_ax,**colorbar_kw)
    
    else:
        #make a comparison plot between data=predictions and targets
        assert len(axes) == 2*len(framelist)
        assert len(targets) == len(data)
        assert fig_kw['ncols'] == 2
        i = 0
        axes = axs
        while i<(fig_kw['nrows']):
            im = axes[i,1].imshow(data[framelist[i]],origin='lower',
                           cmap = colorbar_kw['cmap'],
                           vmin = colorbar_kw['vmin'],
                           vmax = colorbar_kw['vmax'])
            time_num = framelist[i]
            time_str = ''+f'{namelist[i]}'+''
            axes[i,1].set_title(time_str,fontsize='x-large')
            # Now, targets
            im = axes[i,0].imshow(targets[framelist[i]],origin='lower',
                           cmap = colorbar_kw['cmap'],
                           vmin = colorbar_kw['vmin'],
                           vmax = colorbar_kw['vmax'])
            time_num = framelist[i]
            time_str = ''+f'{namelist[i]}'+''
            axes[i,0].set_title(time_str,fontsize='x-large')
            
            if axis_on==False:
                axes[ i,0 ].axis('off') # disable index etc
                axes[i,1].axis('off')
            
            i += 1
            
        del colorbar_kw['cmap'] #not for use in colorbar()
        del colorbar_kw['vmin']
        del colorbar_kw['vmax']
        for i, ax in enumerate(axes.flatten()[:2]):
            if i==0:
                ax.set_title("Targets\n"+f"{namelist[0]}", fontsize='x-large')
                
            if i==1:
                ax.set_title("Predictions\n"+f"{namelist[0]}", fontsize='x-large')

            fig.subplots_adjust(bottom=0.02)
        #[left, bottom, width, height]
        cbar_ax = fig.add_axes([0.125, 0., .775, 0.025])
        cbar = fig.colorbar(im, cax=cbar_ax,**colorbar_kw)
    
    # Save and such.
    if plotname is None:
        plotname = 'multiplot_example.pdf'

    fig.savefig(plotname,bbox_inches='tight',dpi=fig_kw['dpi'], format='pdf')
    return (fig,axes)
