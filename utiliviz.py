"""
UtiliViz

System utilization visualizer to profile compute and memory utilization
of code blocks.
"""
from __future__ import print_function, absolute_import

import multiprocessing as mp
from queue import Empty
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from timeit import default_timer as timer

import psutil


# The interval for the profiler to sample the system.
DEFAULT_INTERVAL = 0.05  # unit: seconds


@contextmanager
def record(mons):
    """A context-manager for containing a profiled code-block.

    Example
    -------

        with utiliviz.record([CpuMon, CudaGpuMon]) as rec:
            code_being_profiled()
        rec.make_plot()             # to make Bokeh plot
        rawdata = rec.get_data()    # to access raw data.
    """
    r = Record(mons)
    r.start()
    try:
        yield r.get_result_obj()
    finally:
        r.stop()


def register_magic(setup_bokeh=False):
    """Use this in Jupyter/IPython notebooks to install the magic command.
    """
    from IPython.core.magic import Magics, magics_class, cell_magic
    from IPython.core.magic_arguments import (magic_arguments, argument,
                                              parse_argstring)
    from bokeh.io import show, output_notebook

    if setup_bokeh:
        output_notebook()

    @magics_class
    class MyMagic(Magics):

        @cell_magic
        @magic_arguments()
        @argument('--nocpu', action='store_true', default=False,
                  help='do not record CPU usage')
        @argument('--cpuoverall', action='store_true', default=False,
                  help='record CPU overall usage')
        @argument('--cuda', action='store_true', default=False,
                  help='record gpu usage')
        def utiliviz(self, line, cell):
            args = parse_argstring(self.utiliviz, line)
            mons = []
            if not args.nocpu:
                if args.cpuoverall:
                    mons.append(CpuOverallMon)
                else:
                    mons.append(CpuMon)
            if args.cuda:
                mons.append(CudaGpuMon)
            raw_code = cell
            r = Record(mons)
            r.start()
            self.shell.run_cell(raw_code, store_history=False)
            r.stop()

            results = r.get_result_obj()
            # It needs more than 2 samples for a good plot
            if len(results) > 2:
                for p in results.make_plot():
                    show(p)
            else:
                print("utiliviz: insufficient sample data for plotting")

    ip = get_ipython()
    ip.register_magics(MyMagic)


def _proc_record(qread, qwrite, monclses, interval):
    mons = [cls() for cls in monclses]
    samples = Samples(mons)
    qwrite.put("STARTED")

    while True:
        try:
            signal = qread.get(True, interval)
        except Empty:
            samples.sample()
        else:
            te = timer()
            assert signal == 'STOP'
            break
    # Get actual start/stop time
    qwrite.put("STOPPED")
    (ts, te) = qread.get()
    sampledresult = samples.finalize(ts, te)
    qwrite.put(sampledresult)
    qwrite.close()
    qread.close()



class Record(object):
    def __init__(self, mons):
        ctx = mp.get_context('spawn')
        self._qwrite = ctx.Queue(1)
        self._qread = ctx.Queue(1)
        interval = DEFAULT_INTERVAL

        proc = ctx.Process(target=_proc_record,
                           args=(self._qwrite,
                                 self._qread,
                                 mons,
                                 interval))
        self._proc = proc

    def get_result_obj(self):
        return RecordedResult(self)

    def start(self):
        self._proc.start()
        got = self._qread.get()
        assert got == 'STARTED'
        self._start_time = timer()

    def stop(self):
        self._stop_time = timer()
        self._qwrite.put('STOP')
        got = self._qread.get()
        assert got == 'STOPPED'
        self._qwrite.put((self._start_time, self._stop_time))
        self.samples = self._qread.get()
        self._qwrite.close()
        self._qread.close()
        self._proc.join()


class RecordedResult(object):
    def __init__(self, record):
        self._record = record

    def __len__(self):
        return len(self._record.samples)

    def get_data(self):
        return self._record.samples.get()

    def make_plot(self):
        """Make a plot of the sample data
        """
        from bokeh.plotting import figure
        from bokeh.models import Legend
        from bokeh import palettes

        def plot_data(name, samples):
            ts, data = zip(*samples)

            ts = [x - ts[0] for x in ts]

            memdata = [d['mem'] for d in data]
            memusedata = defaultdict(list)
            for rec in memdata:
                for k, v in rec.items():
                    memusedata[k].append(v[0] / v[1] * 100)

            compdata = [d['compute'] for d in data]
            computildata = defaultdict(list)
            for rec in compdata:
                for k, v in rec.items():
                    computildata[k].append(v * 100)

            # Make plot
            plot = figure(plot_height=200, title=name,
                          x_range=(0, max(ts)), y_range=(0, 105),
                          background_fill_color='#EEEEEE',
                          toolbar_location=None)
            plot.xaxis.axis_label = 'seconds'
            plot.yaxis.axis_label = 'util %'
            plot.xaxis.axis_label_text_font_size = '8pt'
            plot.yaxis.axis_label_text_font_size = '8pt'
            plot.xgrid.grid_line_color = None
            plot.ygrid.grid_line_color = None
            legend_items = []

            widths = [a - b for a, b in zip(ts[1:], ts[:-1])]
            widths.append(widths[-1])

            cmap = list(palettes.Category10[10])

            def cmap_picker():
                while True:
                    for c in cmap:
                        yield c

            picker = cmap_picker()
            colors = defaultdict(lambda: next(picker))

            for k, vs in memusedata.items():
                ln = plot.rect(x=[t + w / 2 for t, w in zip(ts, widths)],
                               y=[v / 2 for v in vs],
                               width=widths, height=vs, alpha=0.5,
                               line_alpha=0, fill_color=colors[k])
                legend_items.append(('MEM-{}'.format(k), [ln]))

            for k, vs in computildata.items():
                styles = dict(line_color=colors[k], line_width=1,
                              line_alpha=0.7)

                if '*' in k:
                    styles.update(line_width=4, line_color='#333333')
                    ln = plot.line(ts, vs, line_dash='dashed', **styles)
                    # markers = plot.triangle(ts, vs, size=7, alpha=0.9,
                    #                         fill_color=styles['line_color'],
                    #                         line_color=styles['line_color'])
                    legend_items.append(('{}'.format(k), [ln]))
                else:
                    ln = plot.line(ts, vs, **styles)
                    cir = plot.circle(ts, vs, size=styles['line_width'] + 1,
                                      fill_color=colors[k], **styles)
                    legend_items.append(('{}'.format(k), [ln, cir]))

            legend = Legend(items=legend_items,
                            label_text_font_size='6pt',
                            location=(0, 0),
                            orientation='horizontal')
            plot.add_layout(legend, 'below')
            return plot

        return [plot_data(name, sample_data)
                for name, sample_data in self.get_data().items()]


class Samples(object):
    def __init__(self, monitors):
        self._samples = OrderedDict()
        self._mons = monitors
        for m in self._mons:
            self._samples[m.name] = []
        self._ct = 0

    def sample(self):
        ts = timer()
        for mon in self._mons:
            self._samples[mon.name].append((ts, mon.sample()))
        self._ct += 1

    def finalize(self, start_time, stop_time):
        for k in self._samples:
            self._samples[k] = [x for x in self._samples[k]
                                if start_time < x[0] < stop_time]
        return SampledResult(self._ct, self._samples)


class SampledResult(object):
    def __init__(self, nsample, samples):
        self._ct = nsample
        self._samples = samples

    def __len__(self):
        return self._ct

    def get(self):
        return self._samples


class Mon(object):
    def get_memory_info(self):
        """Returns `{"proc_name": (used, total)}`.
        Memory usage is reported in bytes.
        """
        raise NotImplementedError

    def get_processor_util(self):
        """Returns `{"proc_name": util_percent }`.
        Utilization is reported as floating point as percentage.
        """
        raise NotImplementedError

    def sample(self):
        return {'mem': self.get_memory_info(),
                'compute': self.get_processor_util()}


class CpuMon(Mon):
    """Uses psutil
    """
    _primed = False
    _percpu = True
    name = 'cpu'


    def __init__(self):
        if not self._primed:
            # trigger psutil.cpu_percent since the first value is meaningless
            psutil.cpu_percent()
            self._primed = True

    def get_memory_info(self):
        out = OrderedDict()
        vm = psutil.virtual_memory()
        out['sys'] = (vm.used, vm.total)
        return out

    def get_processor_util(self):
        out = OrderedDict()
        if self._percpu:
            values = []
            for i, v in enumerate(psutil.cpu_percent(percpu=self._percpu)):
                v /= 100
                values.append(v)
                out['cpu{}'.format(i)] = v
            n = len(out)
            out['cpu-overall*'] = sum(values) / n
        else:
            out['cpu'] = psutil.cpu_percent(percpu=self._percpu) / 100
        return out


class CpuOverallMon(CpuMon):
    _percpu = False
    name = 'cpu-overall'


class CudaGpuMon(Mon):
    """Uses py3nvml
    """
    name = 'cuda'

    def __init__(self):
        import py3nvml.py3nvml as nvml
        self._nvml = nvml
        self._gpus = OrderedDict()

        self._init()

    def _init(self):
        self._nvml.nvmlInit()
        ngpus = self._nvml.nvmlDeviceGetCount()
        for i in range(ngpus):
            handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
            device_name = self._nvml.nvmlDeviceGetName(handle)
            name = "CUDA {}: {}".format(i, device_name)
            self._gpus[name] = handle

    def get_memory_info(self):
        out = OrderedDict()
        for k, hdl in self._gpus.items():
            info = self._nvml.nvmlDeviceGetMemoryInfo(hdl)
            out[k] = (info.used, info.total)
        return out

    def get_processor_util(self):
        out = OrderedDict()
        for k, hdl in self._gpus.items():
            util = self._nvml.nvmlDeviceGetUtilizationRates(hdl)
            out[k] = util.gpu / 100
        return out
