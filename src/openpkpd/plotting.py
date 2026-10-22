"""Headless scientific plots as portable base64 PNG images (no hosted assets)."""
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def generate_pk_plots(profile: dict | None = None, comparison: dict | None = None, *,
                      output_directory: str | None = None, dpi: int = 120) -> dict:
    """Plot simulation/fit results and/or compare_pk_models output.

    Images contain units, legends and explicit observed/predicted labels. Zero values
    are omitted from the semilog panel and reported. No epsilon is substituted.
    Multiple-dose plots mark administration times; concentrations at bolus times
    are post-dose and connected lines do not depict the instantaneous jump exactly.
    """
    if isinstance(dpi,bool) or not isinstance(dpi,int) or not 72<=dpi<=300:
        raise ValueError('dpi must be an integer between 72 and 300')
    if profile is None and comparison is None:
        raise ValueError('profile or comparison is required')
    images, warnings = [], []
    directory = Path(output_directory) if output_directory is not None else None
    if directory is not None:
        directory.mkdir(parents=True,exist_ok=True)
    def render(name, title, xlabel, ylabel, draw):
        fig = Figure(figsize=(6,4),layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        draw(ax)
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        ax.grid(True,alpha=.25)
        buf = BytesIO()
        fig.savefig(buf,format='png',dpi=dpi)
        data = buf.getvalue()
        item = {'name':name,'mime_type':'image/png','data_base64':base64.b64encode(data).decode('ascii')}
        if directory is not None:
            path = directory/f'{name}.png'
            path.write_bytes(data)
            item['path'] = str(path.resolve())
        images.append(item)
        fig.clear()
    if profile is not None:
        t = np.asarray(profile['time_h'],float)
        fitted = 'predicted_mg_L' in profile
        pred = np.asarray(profile['predicted_mg_L'] if fitted else profile['concentration_mg_L'],float)
        obs = np.asarray(profile['observed_mg_L'],float) if fitted else None
        arrays = [t,pred]+([obs] if fitted else [])
        if any(a.ndim!=1 or len(a)!=len(t) or np.any(~np.isfinite(a)) for a in arrays) or len(t)<2 or np.any(np.diff(t)<=0) or np.any(t<0) or np.any(pred<0) or (fitted and np.any(obs<0)):
            raise ValueError('finite nonnegative ordered time/profile vectors required')
        def profile_draw(ax,log=False):
            mask = pred>0 if log else np.ones(len(t),bool)
            ax.plot(t[mask],pred[mask],label='Predicted' if fitted else 'Simulated')
            if fitted:
                mask_o = obs>0 if log else np.ones(len(t),bool)
                ax.scatter(t[mask_o],obs[mask_o],s=14,label='Observed',color='black')
            for i,dt in enumerate(profile.get('dose_times_h',[])):
                ax.axvline(dt,color='grey',alpha=.25,linestyle=':',label='Administration' if i==0 else None)
            if log: ax.set_yscale('log')
            ax.legend()
        render('concentration_linear','Concentration–time profile','Time (h)','Concentration (mg/L)',profile_draw)
        if np.any(pred>0) or (fitted and np.any(obs>0)):
            render('concentration_semilog','Concentration–time profile (semilog)','Time (h)','Concentration (mg/L, log scale)',lambda ax:profile_draw(ax,True))
        else:
            warnings.append('Semilog plot omitted: no positive concentrations.')
        if np.any(pred==0) or (fitted and np.any(obs==0)):
            warnings.append('Zero concentrations omitted from semilog plot; no substitution was made.')
        if fitted:
            residual = obs-pred
            def residual_draw(ax):
                ax.scatter(t,residual,s=14)
                ax.axhline(0,color='black',linewidth=.8)
            render('residuals_time','Residuals versus time','Time (h)','Observed − predicted (mg/L)',residual_draw)
            def scatter(ax):
                ax.scatter(pred,obs,s=16)
                low, high = min(min(pred),min(obs)),max(max(pred),max(obs))
                ax.plot([low,high],[low,high],color='black',linestyle='--',label='Identity')
                ax.legend()
            render('observed_predicted','Observed versus predicted','Predicted concentration (mg/L)','Observed concentration (mg/L)',scatter)
    if comparison is not None:
        rows = comparison['models']
        if not rows:
            raise ValueError('comparison must contain models')
        labels = [f"{r['model_index']}: {r['n_compartments']}C" for r in rows]
        values = [r.get('delta_aic') for r in rows]
        indices = [i for i,v in enumerate(values) if v is not None and np.isfinite(v)]
        if indices:
            render('comparison_aic','Model comparison (same data and likelihood)','Model index: compartments','ΔAIC',
                   lambda ax:ax.bar([labels[i] for i in indices],[values[i] for i in indices]))
        if len(indices)!=len(rows):
            warnings.append('Models without finite AIC omitted from AIC panel.')
        rmse = [r['rmse_mg_L'] for r in rows]
        if not all(np.isfinite(x) and x>=0 for x in rmse):
            raise ValueError('comparison RMSE must be finite and nonnegative')
        render('comparison_rmse','Model comparison (descriptive error)','Model index: compartments','RMSE (mg/L)',lambda ax:ax.bar(labels,rmse))
        warnings.extend(comparison.get('warnings',[]))
    return {'images':images,'warnings':warnings,'encoding':'base64','format':'PNG',
            'note':'Plots are descriptive and do not establish scientific validity.'}
