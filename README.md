# UtiliViz

System utilization visualizer.

A simple module to support statistic profiling of system utilization.
Currently monitors memory and processor utilization.

Supports NVIDIA GPU via py3nvvml.


## Dependencies


Required:

- Python 3+
- psutil

Optional:

- py3nvml is needed for profiling CUDA GPUs: `pip install py3nvml`


## API
### Use as contextmanager

In regular python, use `record(list_of_monitors)` as a context-manager to
contain the code being profiled:

```python
import utiliviz as uv

with uv.record([uv.CpuMon]) as rec:
    some_code()            # this will be profiled
raw_data = rec.get_data()  # get raw data
plots = rec.make_plot()    # get bokeh plots
```

### Monitors

* The CPU monitor is `utiliviz.CpuMon`.  It checks the system RAM and per-core
  utilization.  The overall CPU utilization is also showed.

* The `utiliviz.CpuOverallMon` is a subset of `utiliviz.CpuMon` that only shows
  the overall CPU utilization.

* CUDA-GPU profiling is enabled by using the `CudaGpuMon` monitor class.


### Use as magic

Provides IPython magic `utiliviz` for Jupyter notebook usage.

The magic must be registered with:

```python
import utiliviz as uv
# setup_bokeh controls whether `bokeh.io.output_notebook()` is called.
uv.register_magic(setup_bokeh=True)
```

Magic cell command usage:

```
    %%utiliviz [--nocpu] [--cuda]
```

