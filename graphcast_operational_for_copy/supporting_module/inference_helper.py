# @title Imports

import dataclasses
import datetime
import functools
import os
import pandas as pd
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
import haiku as hk
import jax
import numpy as np
import xarray


graphcast_weight_path = 'graphcast_weight'

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

# load model choose GraphCast GraphCast_small GraphCast_operational


params_file = graphcast_weight_path + '/params/params_GraphCast_operational.npz'
ckpt = checkpoint.load(params_file, graphcast.CheckPoint)
params = ckpt.params
state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config


diffs_stddev_by_level = xarray.load_dataset(graphcast_weight_path+'/stats/stats_diffs_stddev_by_level.nc').compute()
mean_by_level = xarray.load_dataset(graphcast_weight_path+'/stats/stats_mean_by_level.nc').compute()
stddev_by_level = xarray.load_dataset(graphcast_weight_path+'/stats/stats_stddev_by_level.nc').compute()

## load data
# @title Build jitted functions, and possibly initialize random weights
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


def graphcast_model(input_data, output_folder, fore_hr):

    # make total_batch
    example_batch = xarray.load_dataset(input_data).compute()
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    print('start predict loop......')
    for i in range(np.int_(fore_hr/6)):
        if i==0:
            sample_batch_new = example_batch.copy()*np.nan
            sample_batch_new.coords['datetime'] = (('batch','time'),pd.array(example_batch.coords['datetime'])+datetime.timedelta(hours=12))
            sample_batch_new = sample_batch_new.drop_sel(time=['0 days 06:00:00'])
            input_batch = xarray.concat([example_batch,sample_batch_new], dim='time')
            # example_batch = .drop('toa_incident_solar_radiation')
        else:
        # gain new input batch
            input_batch = input_batch.drop_sel(time=['0 days 00:00:00','0 days 12:00:00'])
            sample_batch_new.coords['datetime'] = (('batch','time'),pd.array(output_predictions.coords['datetime'])+datetime.timedelta(hours=6))
            input_batch = xarray.concat([input_batch,output_predictions,sample_batch_new], dim='time')
            # example_batch = .drop('toa_incident_solar_radiation')
        input_batch.coords['time'] = pd.array(['0 days 00:00:00', '0 days 06:00:00','0 days 12:00:00'], dtype='timedelta64[ns]')
        input_batch['geopotential_at_surface'] = (('lat', 'lon'), np.array(example_batch.variables['geopotential_at_surface']))
        input_batch['land_sea_mask'] = (('lat', 'lon'), np.array(example_batch.variables['land_sea_mask']))

        eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
            input_batch, target_lead_times=slice("6h", f"{1*6}h"),
            **dataclasses.asdict(task_config))

        print('start predict rollout......')
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=eval_inputs,
            targets_template=eval_targets * np.nan,
            forcings=eval_forcings)

        # save 6hr predict data
        output_predictions = sample_batch_new.copy()
        output_predictions.coords['datetime'] = (('batch','time'),pd.array(sample_batch_new.coords['datetime']))
        output_predictions['2m_temperature'] = (("batch", "time", "lat", "lon"), np.array(predictions.variables['2m_temperature']))
        output_predictions['mean_sea_level_pressure'] = (("batch", "time", "lat", "lon"), np.array(predictions.variables['mean_sea_level_pressure']))
        output_predictions['10m_u_component_of_wind'] = (("batch", "time", "lat", "lon"), np.array(predictions.variables['10m_u_component_of_wind']))
        output_predictions['10m_v_component_of_wind'] = (("batch", "time", "lat", "lon"), np.array(predictions.variables['10m_v_component_of_wind']))
        output_predictions['total_precipitation_6hr'] = (("batch", "time", "lat", "lon"), np.array(predictions.variables['total_precipitation_6hr']))
        output_predictions['temperature'] = (("batch", "time", "level", "lat", "lon"), np.array(predictions.variables['temperature']))
        output_predictions['geopotential'] = (("batch", "time", "level", "lat", "lon"), np.array(predictions.variables['geopotential']))
        output_predictions['u_component_of_wind'] = (("batch", "time", "level", "lat", "lon"), np.array(predictions.variables['u_component_of_wind']))
        output_predictions['v_component_of_wind'] = (("batch", "time", "level", "lat", "lon"), np.array(predictions.variables['v_component_of_wind']))
        output_predictions['vertical_velocity'] = (("batch", "time", "level", "lat", "lon"), np.array(predictions.variables['vertical_velocity']))
        output_predictions['specific_humidity'] = (("batch", "time", "level", "lat", "lon"), np.array(predictions.variables['specific_humidity']))
        output_predictions['geopotential_at_surface'] = (('lat', 'lon'), np.array(example_batch.variables['geopotential_at_surface']))
        output_predictions['land_sea_mask'] = (('lat', 'lon'), np.array(example_batch.variables['land_sea_mask']))




        save_path = f'{output_folder}/gc_operational_predict_data_{i+1}.nc'
        output_predictions.to_netcdf(save_path)
        print(f'finish predict {(i+1)*6}hr')


    


