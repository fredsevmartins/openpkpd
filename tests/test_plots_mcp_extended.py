import asyncio
import base64
import io
import json
import os
from pathlib import Path
import struct
import numpy as np
import pytest
from mcp import Client, StdioServerParameters
from openpkpd import generate_pk_plots, simulate_pk, simulate_multiple_doses, fit_pk_model, compare_pk_models


def fit_sample():
    t=np.linspace(0,24,20)
    return fit_pk_model(t,2*np.exp(-.1*t)+.01*np.cos(t),100,{'CL':4,'Vc':45})


def test_plot_outputs_are_portable_real_pngs(tmp_path):
    fit=fit_sample()
    comparison=compare_pk_models([fit,fit])
    result=generate_pk_plots(fit,comparison,output_directory=str(tmp_path))
    assert {x['name'] for x in result['images']} == {'concentration_linear','concentration_semilog',
            'residuals_time','observed_predicted','comparison_aic','comparison_rmse'}
    for img in result['images']:
        data=base64.b64decode(img['data_base64'],validate=True)
        assert data[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',data[16:24])
        assert width>=600 and height>=400
        assert Path(img['path']).read_bytes()==data
    # Matplotlib's image reader independently decodes complete pixel content.
    from matplotlib.image import imread
    pixels=imread(io.BytesIO(data),format='png')
    assert pixels.std()>.05


def test_multiple_doses_and_zero_log_policy():
    result=generate_pk_plots(simulate_multiple_doses(100,5,50,12,3,route='oral',ka_per_h=1))
    assert len(result['images'])==2
    assert any('Zero' in x for x in result['warnings'])
    zeros=generate_pk_plots({'time_h':[0,1], 'concentration_mg_L':[0,0]})
    assert len(zeros['images'])==1
    with pytest.raises(ValueError): generate_pk_plots()
    with pytest.raises(ValueError): generate_pk_plots(simulate_pk(100,5,50),dpi=0)


def test_all_mcp_tools_structured_results_and_images():
    root=Path(__file__).resolve().parents[1]
    async def check():
        params=StdioServerParameters(command=str(root/'.venv/bin/openpkpd-mcp'),args=[],cwd=root,
                                     env={**os.environ,'PYTHONPATH':str(root/'src')})
        async with Client(params) as client:
            async def call(name,args):
                r=await client.call_tool(name,args)
                assert not r.is_error, str(r)
                assert r.structured_content is not None
                # Strict JSON catches NaN/Infinity, which are not portable JSON values.
                json.dumps(r.structured_content,allow_nan=False)
                return r
            sim=(await call('simulate_pk',{'dose_mg':100,'clearance_L_h':5,'volume_L':50,'n_points':11})).structured_content
            assert sim['aucinf_mg_h_L']==20
            nca=(await call('calculate_nca',{'time_h':[0,1,2,3,4,5],
                         'concentration_mg_L':np.exp(-np.arange(6.)).tolist(), 'integration_method':'linear_up_log_down'})).structured_content
            assert nca['lambda_z_per_h']==pytest.approx(1)
            pd=(await call('simulate_pd',{'concentrations_mg_L':[2],'emax':10,'ec50_mg_L':2})).structured_content
            assert pd['effect']==[5]
            multiple=(await call('simulate_multiple_doses',{'dose_mg':100,'clearance_L_h':5,'volume_L':50,
                                  'dose_interval_h':12,'n_doses':3,'points_per_interval':11})).structured_content
            assert multiple['dose_times_h']==[0,12,24]
            t=np.linspace(0,24,20)
            fit=(await call('fit_pk_model',{'time_h':t.tolist(),'concentration_mg_L':(2*np.exp(-.1*t)+.01*np.cos(t)).tolist(),
                           'dose_mg':100,'initial_parameters':{'CL':4,'Vc':45}})).structured_content
            assert fit['convergence']['success']
            diagnostic=(await call('diagnose_pk_fit',{'fit':fit})).structured_content
            assert diagnostic['parameter_plausibility']['positive_clearances_volumes_rates']
            comparison=(await call('compare_pk_models',{'fits':[fit,fit]})).structured_content
            assert len(comparison['models'])==2
            images=await call('generate_pk_plots',{'profile':fit,'comparison':comparison,'dpi':72})
            assert sum(c.type=='image' for c in images.content)==6
            for block in images.content:
                if block.type=='image': assert base64.b64decode(block.data).startswith(b'\x89PNG')
            invalid=await client.call_tool('simulate_pk',{'dose_mg':-1,'clearance_L_h':5,'volume_L':50})
            assert invalid.is_error
    asyncio.run(check())
